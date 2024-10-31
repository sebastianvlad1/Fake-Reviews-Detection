import torch
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
from config import TOKENIZER_PATH, DEVICE

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256)

class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(DEVICE)
        return item

    def __len__(self):
        return len(self.labels)
