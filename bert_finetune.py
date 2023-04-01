import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split


class NewsletterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_ids = self.data[index]['input_ids']
        attention_mask = self.data[index]['attention_mask']
        proportion = self.data[index]['proportion']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'proportion': torch.tensor(proportion, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.data)

class NewsletterPredictor(torch.nn.Module):
    def __init__(self, num_classes):
        super(NewsletterPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, num_classes)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.view(-1))
        else:
            loss = None
        return logits, loss
    
def tokenize_data(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_data = tokenizer(data['title'] + ' ' + data['headlines'], 
                               padding='max_length', 
                               truncation=True, 
                               max_length=512, 
                               return_tensors='pt')
    tokenized_data['proportion'] = data['proportion']
    return tokenized_data

def finetune_bert_model() -> None:
    newsletter_file = open("b_json/newsletters_full_data.json", "r")
    newsletter_data = json.load(newsletter_file)
    newsletter_file.close()
    data = pd.DataFrame.from_dict(newsletter_data, orient='index')

    #create 70%, 15%, 15% train, validation and test split
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data = train_data.apply(tokenize_data, axis=1)
    val_data = val_data.apply(tokenize_data, axis=1)
    test_data = test_data.apply(tokenize_data, axis=1)


    batch_size = 16

    train_dataset = NewsletterDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = NewsletterDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = NewsletterDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NewsletterPredictor(num_classes=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    train_losses = []
    val_losses = []



    num_epochs = 50
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].squeeze(dim=1).to(device)
            attention_mask = batch['attention_mask'].squeeze(dim=1).to(device)
            proportions = batch['proportion'].to(device)

            optimizer.zero_grad()
            outputs, loss= model(input_ids, attention_mask, proportions)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
        train_loss = train_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        #Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].squeeze(dim=1).to(device)
                attention_mask = batch['attention_mask'].squeeze(dim=1).to(device)
                proportions = batch['proportion'].to(device)

                outputs, loss = model(input_ids, attention_mask, proportions)

                val_loss += loss.item() * input_ids.size(0)

        val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, ")

    test_loss = 0

    # loop through the test set batches
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].squeeze(dim=1).to(device)
            attention_mask = batch['attention_mask'].squeeze(dim=1).to(device)
            proportions = batch['proportion'].to(device)

            outputs, loss = model(input_ids, attention_mask, proportions)

            test_loss += loss.item() * input_ids.size(0)

    # calculate the average loss over all the batches
    test_loss /= len(test_dataloader.dataset)

    # print the results
    print(f'Test Loss: {test_loss:.6f}')