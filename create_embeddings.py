from transformers import BertModel, BertTokenizer
import torch


def create_embeddings(email_subjects, email_contents):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    labels = [1, 0]
    
    preprocessed_subjects = [text.lower() for text in email_subjects]
    preprocessed_contents = [text.lower() for text in email_contents]
    
    subject_tokens = [tokenizer.encode(text, add_special_tokens=True) for text in preprocessed_subjects]
    content_tokens = [tokenizer.encode(text, add_special_tokens=True) for text in preprocessed_contents]
    
    subject_embeddings = []
    content_embeddings = []

    for tokens in subject_tokens:
        input_ids = torch.tensor(tokens).unsqueeze(0)
        outputs = bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_state, dim=1)
        subject_embeddings.append(mean_embedding.detach().numpy()[0])

    for tokens in content_tokens:
        input_ids = torch.tensor(tokens).unsqueeze(0)
        outputs = bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_state, dim=1)
        content_embeddings.append(mean_embedding.detach().numpy()[0])
    
    return subject_embeddings, content_embeddings


    
    