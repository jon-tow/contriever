# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
NOTE:
- We remove the option to normalize representations thereby removing `norm_query`
and `norm_doc` attributes in the MoCo class. It complicates? representation gradient
caching and is off by default in the original code.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import copy
import transformers

from contextlib import nullcontext
from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)

class MoCo(nn.Module):
    def __init__(self, opt):
        super(MoCo, self).__init__()

        self.queue_size = opt.queue_size
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k #apply the encoder on keys in train mode
        self.micro_batch_size = opt.micro_batch_size

        retriever, tokenizer = self._load_retriever(opt)
        
        self.tokenizer = tokenizer

        # Turn this off while testing
        if dist.is_initialized():
            self.encoder_q = torch.nn.parallel.DistributedDataParallel(
                retriever.cuda(),
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            self.encoder_k = torch.nn.parallel.DistributedDataParallel(
                copy.deepcopy(retriever).cuda(),
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=False,
            )
            dist.barrier()
        else:
            self.encoder_q = retriever
            self.encoder_k = copy.deepcopy(retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

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

    def _split_inputs(self, inputs, size):
        return list(inputs.split(size, dim=0))

    def _get_query_reprs(self, inputs, masks):
        """Forward pass without gradient computation to record representations."""
        reprs = []
        with torch.no_grad():
            for input, mask in zip(inputs, masks):
                repr = self.encoder_q(input_ids=input, attention_mask=mask)
                reprs.append(repr)
        reprs = torch.cat(reprs, dim = 0)
        return reprs

    def _get_key_reprs(self, inputs, mask):
        """Returns the representations of the keys."""
        reprs = []
        with torch.no_grad():
            self._momentum_update_key_encoder()  # Update the key encoder
            for input, mask in zip(inputs, mask):
                if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                    self.encoder_k.eval()
                repr = self.encoder_k(input_ids=input, attention_mask=mask)
                reprs.append(repr)
        reprs = torch.cat(reprs, dim = 0)
        return reprs

    def all_gather(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix='', **kwargs):
        """Performs a gradient cached training step."""
        iter_stats = {}
        batch_size = q_tokens.shape[0]
        # print("="*80)
        # print("batch_size:", batch_size)
        # print(f"q_tokens shape: {q_tokens.shape}")
        # print(f"k_tokens shape: {k_tokens.shape}")
        # 0. Divide query and key tokens into sets of micro-batches each of which
        # can fit into memory for gradient computation.
        # shape: [batch_size, micro_batch_size, q_tokens_len]
        q_micro_tokens = self._split_inputs(q_tokens, self.micro_batch_size)
        q_micro_masks = self._split_inputs(q_mask, self.micro_batch_size)
        # print(f"q_micro_tokens: {len(q_micro_tokens)}")
        # print(f"q_micro_tokens shape: {q_micro_tokens[0].shape}")

        # shape: [batch_size, micro_batch_size, k_tokens_len]
        k_micro_tokens = self._split_inputs(k_tokens, self.micro_batch_size)
        k_micro_masks = self._split_inputs(k_mask, self.micro_batch_size)
        # print(f"k_micro_tokens: {len(k_micro_tokens)}")
        # print(f"K_micro_tokens shape: {k_micro_tokens[0].shape}")

        # 1. Graph-less Forward: Run an extra encoder forward pass for each batch
        # instance to get its representation/embedding. (NO GRAPH CONSTUCTION)
        q_reprs = self._get_query_reprs(q_micro_tokens, q_micro_masks)
        print(f"q_reprs shape: {q_reprs.shape}")
        k_reprs = self._get_key_reprs(k_micro_tokens, k_micro_masks)
        # print(f"k_reprs shape: {k_reprs.shape}")

        # 1a. When training on multi-GPU; we need to compute the gradients with
        # all examples across all GPUs. This requires an all-gather to make
        # representations available on all GPUs.
        if dist.is_initialized():
            print("All-gather-ing...")
            q_reprs = self.all_gather(q_reprs)
            k_reprs = self.all_gather(k_reprs)
        print(f"q_reprs shape: {q_reprs.shape}")

        # 2. Representation Gradient Computation and Caching: Run backward-pass
        # to populate gradients for the query representations (δL / δencoder_q).
        # NOTE: The encoders are not included in this gradient computation b/c the graph-less forward.
        q_reprs = q_reprs.detach().requires_grad_()  # Remove `q_reprs` from encoder_q op graph.
        logits = self._compute_logits(q_reprs, k_reprs) / self.temperature
        print(f"logits: {logits.shape}")
        # print(f"logits: {logits}")
        labels = torch.zeros(q_reprs.size(0), dtype=torch.long).cuda()  # Positives are the 0-th
        print(f"labels: {labels.shape}")
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss.backward()
        q_reprs_grad_cache = self._split_inputs(q_reprs.grad, self.micro_batch_size)
        loss = loss.detach()
        # print(f"loss: {loss}")

        # 3. Sub-batch Gradient Accumulation: Run the second forward and backward
        # pass to compute gradient for the query encoder model.
        for tokens, mask, repr_grad in zip(
            q_micro_tokens, q_micro_masks, q_reprs_grad_cache
        ):
            # Run encoder forward one sub-batch at a time to compute representations
            # and build the corresponding computation graph.
            repr = self.encoder_q(input_ids=tokens, attention_mask=mask)
            # Take the micro-batch’s representation gradients from the cache and
            # run back-prop through the query encoder model.
            surrogate = torch.dot(repr.flatten(), repr_grad.flatten())
            surrogate.backward()

        self._dequeue_and_enqueue(k_reprs)

        # Get stats for this step.
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + '/'
        iter_stats[f'{stats_prefix}loss'] = (loss.item(), batch_size)
        predicted_idx = torch.argmax(logits, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(q_reprs, dim=0).mean().item()
        stdk = torch.std(k_reprs, dim=0).mean().item()
        iter_stats[f'{stats_prefix}accuracy'] = (accuracy, batch_size)
        iter_stats[f'{stats_prefix}stdq'] = (stdq, batch_size)
        iter_stats[f'{stats_prefix}stdk'] = (stdk, batch_size)

        print("="*80)
        return loss, iter_stats
