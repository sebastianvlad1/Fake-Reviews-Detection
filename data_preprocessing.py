import pandas as pd
import re
from sklearn.model_selection import train_test_split
from config import TRAIN_DATA_PATH

def load_and_prepare_data():
    df = pd.read_csv(TRAIN_DATA_PATH).drop(['id'], axis=1)
    df['label'] = df['label'].map({'CG': 1, 'OR': 0}).astype(int)
    df['review'] = df['review'].apply(clean_text)
    return df

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

def split_data(df):
    X = df["review"].tolist()
    Y = df["label"].tolist()
    return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
