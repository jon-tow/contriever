# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import copy
import transformers

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)

class MoCo(nn.Module):
    def __init__(self, opt):
        super(MoCo, self).__init__()

        self.queue_size = opt.queue_size
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k #apply the encoder on keys in train mode
        self.micro_batch_size = opt.micro_batch_size

        retriever, tokenizer = self._load_retriever(opt)
        
        self.tokenizer = tokenizer
        self.encoder_q = retriever
        self.encoder_k = copy.deepcopy(retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        # create the queue
        self.register_buffer("queue", torch.randn(opt.projection_size, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _load_retriever(self, opt):
        model_id, tokenizer_id = opt.retriever_model_id, opt.retriever_tokenizer_id
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, tokenizer_id)

        if opt.random_init:
            retriever = contriever.Contriever(cfg)
        else:
            retriever = utils.load_hf(contriever.Contriever, model_id)

        if 'bert-' in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"
            if 'bit' in opt.optim:
                import bitsandbytes as bnb
                retriever.embeddings.word_embeddings = bnb.nn.StableEmbedding(
                    retriever.config.vocab_size, retriever.config.hidden_size)

        retriever.config.pooling = opt.pooling

        return retriever, tokenizer

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f'{batch_size}, {self.queue_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def _compute_logits(self, q, k):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) 

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits

    def _split_inputs(self, inputs: torch.Tensor, size: int) -> torch.Tensor:
        return list(inputs.split(size, dim=0))

    def _get_query_reprs(self, encoder, inputs, masks, use_normalize):
        """Forward pass without gradient computation to record representations."""
        reprs = []
        with torch.no_grad():
            for input, mask in zip(inputs, masks):
                repr = encoder(input_ids=input, attention_mask=mask)
                if use_normalize:
                    repr = nn.functional.normalize(repr, dim=1)
                reprs.append(repr)
        reprs = torch.cat(reprs, dim = 0)
        return reprs

    def _get_key_reprs(self, encoder, inputs, mask, use_normalize):
        """Returns the representation of the keys."""
        reprs = []
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            for input, mask in zip(inputs, mask):
                if not encoder.training and not self.moco_train_mode_encoder_k:
                    encoder.eval()
                repr = encoder(input_ids=input, attention_mask=mask)  # keys: NxC
                if use_normalize:
                    repr = nn.functional.normalize(repr, dim=-1)
                reprs.append(repr)
        reprs = torch.cat(reprs, dim = 0)
        return reprs

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix='', **kwargs):
        """Performs a gradient cached training step."""
        iter_stats = {}
        batch_size = q_tokens.shape[0]

        # 0. Divide query and key tokens into a set of sub-batches each of which
        # can fit into memory for gradient computation.
        q̂_tokens = self._split_inputs(q_tokens, self.micro_batch_size)
        q̂_mask = self._split_inputs(q_mask, self.micro_batch_size)
        k̂_tokens = self._split_inputs(k_tokens, self.micro_batch_size)
        k̂_mask = self._split_inputs(k_mask, self.micro_batch_size)

        # 1. Graph-less Forward: Run an extra encoder forward pass for each batch
        # instance to get its representation. Collect and store.
        q̂_reprs = self._get_query_reprs(
            self.encoder_q, q̂_tokens, q̂_mask, self.norm_query)
        k̂_reprs = self._get_key_reprs(
            self.encoder_k, k̂_tokens, k̂_mask, self.norm_doc)

        # 2. Representation Gradient Computation and Caching: Run backward-pass
        # to populate gradients for each representation.
        # NOTE: The encoders are not included in this gradient computation.
        # because of the graph-less forward.
        # NOTE: Only need the gradients for the query representations.
        q̂_reprs = q̂_reprs.detach().requires_grad_()  # [repr.detach().requires_grad_() for repr in q̂_reprs], dim=0
        logits = self._compute_logits(q̂_reprs, k̂_reprs) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()  # positives are the 0-th
        # TODO: Possibly add `with autocast():`
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss.backward()
        q̂_repr_grad_cache = self._split_inputs(q̂_reprs.grad, self.micro_batch_size)
        loss = loss.detach()

        # 3. Sub-batch Gradient Accumulation: 
        for x, mask, repr_grad in zip(q̂_tokens, q̂_mask, q̂_repr_grad_cache):
            # Run encoder forward one sub-batch at a time to compute representations
            # and build the corresponding computation graph. 
            repr = self.encoder_q(input_ids=x, attention_mask=mask)
            if self.norm_query:
                repr = nn.functional.normalize(repr, dim=1)
            # Take the micro-batch’s representation gradients from the cache and
            # run back-prop through the (query) encoder.
            surrogate = torch.dot(repr.flatten(), repr_grad.flatten())
            surrogate.backward()

        # Get stats for this step.
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + '/'
        iter_stats[f'{stats_prefix}loss'] = (loss.item(), batch_size)
        predicted_idx = torch.argmax(logits, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(q̂_reprs, dim=0).mean().item()
        stdk = torch.std(k̂_reprs, dim=0).mean().item()
        iter_stats[f'{stats_prefix}accuracy'] = (accuracy, batch_size)
        iter_stats[f'{stats_prefix}stdq'] = (stdq, batch_size)
        iter_stats[f'{stats_prefix}stdk'] = (stdk, batch_size)

        self._dequeue_and_enqueue(k̂_reprs)

        return loss, iter_stats
           
    # def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix='', **kwargs):
    #     iter_stats = {}
    #     batch_size = q_tokens.size(0)

    #     q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask) # queries: NxC
    #     if self.norm_query:
    #         q = nn.functional.normalize(q, dim=1)

    #     # compute key features
    #     with torch.no_grad():  # no gradient to keys
    #         self._momentum_update_key_encoder()  # update the key encoder

    #         if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
    #             self.encoder_k.eval()

    #         k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: NxC
    #         if self.norm_doc:
    #             k = nn.functional.normalize(k, dim=-1)

    #     logits = self._compute_logits(q, k) / self.temperature

    #     # labels: positive key indicators
    #     labels = torch.zeros(batch_size, dtype=torch.long).cuda()

    #     loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # if len(stats_prefix) > 0:
        #     stats_prefix = stats_prefix + '/'
        # iter_stats[f'{stats_prefix}loss'] = (loss.item(), batch_size)

        # predicted_idx = torch.argmax(logits, dim=-1)
        # accuracy = 100 * (predicted_idx == labels).float().mean()
        # stdq = torch.std(q, dim=0).mean().item()
        # stdk = torch.std(k, dim=0).mean().item()
        # iter_stats[f'{stats_prefix}accuracy'] = (accuracy, batch_size)
        # iter_stats[f'{stats_prefix}stdq'] = (stdq, batch_size)
        # iter_stats[f'{stats_prefix}stdk'] = (stdk, batch_size)

        # self._dequeue_and_enqueue(k)

        # return loss, iter_stats
