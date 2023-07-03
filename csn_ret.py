"""Author: @Reshinth from CarperAI"""
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel
from typing import *


def load_dataset_for_eval(
    dataset_id: str,
    subset: str,
    candidate_size: Optional[str] = 1000,
    seed: Optional[int] = 26037,
):
    """
    Loads a dataset from the hf datasets library and corresponding candidates.
    :param dataset_idt: dataset id corresponding to the dataset.
    :param subset: subset of the dataset to evaluate.
    :param candidate_size: size of the candidate space.
    """
    dataset = load_dataset(dataset_id, subset, split="test", streaming=True)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.take(candidate_size)
    return dataset


def write_eval_dataset(
    dataset_id: str,
    subset: str,
    output_dir: str,
    candidate_size: Optional[int] = 10,
    split: Optional[str] = "test",
    seed: Optional[int] = 2601934,
):
    import gzip
    import jsonlines
    import shutil
    dataset = load_dataset(dataset_id, subset, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.take(candidate_size)
    dataset = list(dataset)
    file_name = f"{output_dir}/code_search_net_{subset}_shuffled.jsonl"
    with jsonlines.open(file_name, "w") as writer:
        writer.write_all(dataset)
    with open(file_name, 'rb') as f_in:
        with gzip.open(f'{file_name}.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def squeeze_tree(tensor_data):
    return {k: tensor_data[k].squeeze(0) for k in tensor_data}


class RetDataset(Dataset):
    def __init__(self, dataset, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataset_list = list(dataset)

    def __len__(self):
        return len(list(self.dataset))

    def __getitem__(self, index: int):
        datapoint = self.dataset_list[index]
        code, nl_query_desc = datapoint["whole_func_string"], datapoint["func_documentation_string"]
        #BUG(reshinth) : Removing the docstring in the function string
        code = code.replace(nl_query_desc,"") 
        code = squeeze_tree(tokenizer(
            code, padding="max_length", truncation=True, return_tensors='pt'))
        nl_query_desc = squeeze_tree(tokenizer(
            nl_query_desc, padding="max_length", truncation=True, return_tensors='pt'))
        return code, nl_query_desc


def mean_pooling(token_embeds, mask):
    token_embeds, mask = token_embeds.detach().cpu(), mask.detach().cpu()
    token_embeds = token_embeds.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeds = token_embeds.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeds


def mrr(scores, accelerator):
    accelerator.print(f"\nScores Shape: {scores.shape}")
    accelerator.print(f"\nScores:\n{scores}")
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1 / rank)
    mrr = float(np.mean(ranks))
    return mrr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/graphcodebert-base")
    parser.add_argument("--dataset", type=str, default="code_search_net")
    parser.add_argument("--subsets", type=str, default="python")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--candidate_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=312026)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    candidate_size = args.candidate_size
    subsets = args.subsets.split(",")

    accelerator = Accelerator()
    device = accelerator.device

    accelerator.print(f"Loading Model:   `{model_name}`")
    accelerator.print(f"Loading Dataset: `{dataset}`")
    accelerator.print(f"Dataset subsets:  `{subsets}`")
    accelerator.print(f"Candidate Size:  {candidate_size}")
    accelerator.print(f"Batch Size:      {batch_size}")

    results = {}

    for subset in subsets:
        test_dataset = load_dataset_for_eval(
            dataset,
            subset,
            candidate_size,
            seed=args.seed
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        test_torch_dataset = RetDataset(test_dataset, tokenizer)
        test_dataloader = DataLoader(test_torch_dataset, batch_size=batch_size)

        accelerator.print(f"\nLoading the model into the memory.\n")
        # if args.model = "microsoft/codereviewer" or "Salesforce/codet5" in args.model:
        #     model = transformers.T5EncoderModel.from_pretrained(args.model)
        # else:
        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=True,
        )

        # Loading model to accelerator
        model, test_dataload = accelerator.prepare(model, test_dataloader)
        ranks, code_embeds_list, nl_query_embeds_list = [], [], []
        for batch in tqdm(test_dataloader, total=int(candidate_size/args.batch_size)):
            code, nl_query_desc = batch[0], batch[1]
            code_outputs = model(
                input_ids=code["input_ids"].to(accelerator.device),
                attention_mask=code["attention_mask"].to(accelerator.device),
                return_dict=True)
            code_embeds = mean_pooling(
                code_outputs['last_hidden_state'],
                code['attention_mask'])
            code_embeds_list.append(code_embeds.cpu().detach())

            nl_query_outputs = model(
                input_ids=nl_query_desc["input_ids"].to(accelerator.device),
                attention_mask=nl_query_desc["attention_mask"].to(accelerator.device),
                return_dict=True)
            nl_query_embeds = mean_pooling(
                nl_query_outputs['last_hidden_state'],
                nl_query_desc['attention_mask'])
            nl_query_embeds_list.append(nl_query_embeds.cpu().detach())

        del model
        accelerator.print("\nCompleted encoding to space...")
        
        nl_query_embeds = torch.cat(nl_query_embeds_list, dim=0).cpu().detach().numpy()
        code_embeds = torch.cat(code_embeds_list, dim=0).cpu().detach().numpy()
        scores = np.matmul(nl_query_embeds, code_embeds.T)

        # Compute evals
        result = mrr(scores, accelerator)
        format_result = f"{(round(result, 4) * 100.0):.2f}"
        accelerator.print(f"\nEval MRR: {format_result}\n")
        results[subset] = format_result

    output_file_name = f"{args.output_dir}/model={args.model.replace('/','_')}.size={candidate_size}.seed={args.seed}.json"
    with open(output_file_name, "w") as f:
        json.dump(results, f)
