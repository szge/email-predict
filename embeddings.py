from transformers import BertModel, BertTokenizer
import torch
import json


def create_embedding(newsletter_title: str, newsletter_headlines: str):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    preprocessed_title = newsletter_title.lower()
    preprocessed_headlines = newsletter_headlines.lower()
    
    title_tokens = tokenizer.encode(preprocessed_title, add_special_tokens=True)
    headline_tokens = tokenizer.encode(preprocessed_headlines, add_special_tokens=True)

    input_ids = torch.tensor(title_tokens + headline_tokens).unsqueeze(0)
    outputs = bert_model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    mean_embedding = torch.mean(last_hidden_state, dim=1)
    bert_embedding = mean_embedding.detach().numpy()[0]

    return bert_embedding

def generate_newspaper_embeddings():
    newsletter_data_file = open("b_json/newsletters.json", "r")
    newsletters = json.load(newsletter_data_file)
    newsletter_data_file.close()
    
    newsletter_embeddings = {}
    
    for id in newsletters:
        title = newsletters[id]["title"]
        headlines = newsletters[id]["headlines"]
        newsletter_embeddings[id] = create_embedding(title, headlines)
    
    return newsletter_embeddings
        


    
    
    