import torch

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

sentences = [
    "How many people live in London?",
    "Around 9 Million people live in London",
    "London is known for its financial district",
]

def test_carptriever1():
    from transformers import AutoTokenizer, AutoModel

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    model_path = "jon-tow/carptriever-1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    model = AutoModel.from_pretrained(model_path, add_pooling_layer=False, use_auth_token=True)

    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Encode sentences 
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

    # Compute dot-product scores between the query and sentence embeddings
    query_embedding, sentence_embeddings = embeddings[0], embeddings[1:]
    scores = (query_embedding @ sentence_embeddings.transpose(0, 1)).cpu().tolist()

    sentence_score_pairs = sorted(zip(sentences[1:], scores), reverse=True)

    print(f"Query: {sentences[0]}")
    for sentence, score in sentence_score_pairs:
        print(f"\nSentence: {sentence}\nScore: {score:.4f}")

def test_carptriever():
    from transformers import AutoTokenizer, AutoModel

    model_path = "jon-tow/carptriever"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    model = AutoModel.from_pretrained(model_path, add_pooling_layer=False, use_auth_token=True)

    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Encode sentences 
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

    # Compute dot-product scores between the query and sentence embeddings
    query_embedding, sentence_embeddings = embeddings[0], embeddings[1:]
    scores = (query_embedding @ sentence_embeddings.transpose(0, 1)).cpu().tolist()

    sentence_score_pairs = sorted(zip(sentences[1:], scores), reverse=True)

    print(f"Query: {sentences[0]}")
    for sentence, score in sentence_score_pairs:
        print(f"\nSentence: {sentence}\nScore: {score:.4f}")

def test_contriever():
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')

    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Encode sentences 
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

    # Compute dot-product scores between the query and sentence embeddings
    query_embedding, sentence_embeddings = embeddings[0], embeddings[1:]
    scores = (query_embedding @ sentence_embeddings.transpose(0, 1)).cpu().tolist()

    sentence_score_pairs = sorted(zip(sentences[1:], scores), reverse=True)

    print(f"Query: {sentences[0]}")
    for sentence, score in sentence_score_pairs:
        print(f"\nSentence: {sentence}\nScore: {score:.4f}")



if __name__ == "__main__":
    print('=' * 80)
   
    print('test_carptriever1')
    test_carptriever1()

    print('=' * 80)
   
    print('test_carptriever')
    test_carptriever()
    
    print('=' * 80)

    print('test_contriever')
    test_contriever()

    print('=' * 80)