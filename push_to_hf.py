import os
import torch
import transformers
from transformers import BertModel

from src import utils
from src import contriever


def push_retriever(model_path):
    #try: #check if model is in a moco wrapper
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict['opt']
        retriever_model_id = opt.retriever_model_id
        print('retriever_model_id:', retriever_model_id)
        retriever_tokenizer_id = opt.retriever_tokenizer_id
        print('retriever_tokenizer_id:', retriever_tokenizer_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_tokenizer_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        retriever = contriever.Contriever(cfg)
        pretrained_dict = pretrained_dict["model"]
        if any("encoder_q." in key for key in pretrained_dict.keys()):
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q" in k}
        retriever.load_state_dict(pretrained_dict)
        # print(retriever)
        retriever.push_to_hub("CarperAI/carptriever-1")
        tokenizer.push_to_hub("CarperAI/carptriever-1")
    else:
        raise Exception("Model not found")
    return retriever, tokenizer


def save_retriever(model_path, repo_path):
    """Saves model to the HF repository path which can then be committed and pushed from."""
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict['opt']
        retriever_model_id = opt.retriever_model_id
        print('retriever_model_id:', retriever_model_id)
        retriever_tokenizer_id = opt.retriever_tokenizer_id
        print('retriever_tokenizer_id:', retriever_tokenizer_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_tokenizer_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        retriever = contriever.Contriever(cfg)
        pretrained_dict = pretrained_dict["model"]
        if any("encoder_q." in key for key in pretrained_dict.keys()):
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q" in k}
        retriever.load_state_dict(pretrained_dict)
        retriever.save_pretrained(repo_path)
        tokenizer.save_pretrained(repo_path)
    else:
        raise Exception("Model not found")
    return retriever, tokenizer


if __name__ == "__main__":
    model_path = "/fsx/carper/contriever/checkpoint/pile_deduped/baseline-deduped-16886-average-adamw-bs64-smooth0.0-rmin0.05-rmax0.5-T0.05-8192-0.999-bert-large-uncased-delete-0.1/checkpoint/step-150000"
    repo_path = "/fsx/carper/contriever/carptriever-1"
    save_retriever(model_path=model_path, repo_path=repo_path)