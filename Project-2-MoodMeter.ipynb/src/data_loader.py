import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        words = text.split()
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

        if len(indices) < self.max_length:
            indices = indices + [self.vocab["<PAD>"]] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def download_imdb_dataset(data_dir="../data"):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'imdb_reviews.csv')

    if not os.path.exists(csv_path):
        print("Downloading IMDB dataset...")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        try:
            from datasets import load_dataset
            dataset = load_dataset("stanfordnlp/imdb")

            df_train = pd.DataFrame(dataset['train'])
            df_test = pd.DataFrame(dataset['test'])
            
            df = pd.concat([df_train, df_test], ignore_index=True)
            df = df[['text', 'label']]
            df.to_csv(csv_path, index=False)
            print(f"Dataset saved to {csv_path}")

        except Exception as e:
            print(f"Error downloading: {e}")
            print("Creating sample dataset...")

            df = pd.DataFrame({
                'text': [
                    "This movie was amazing! I loved it.",
                    "Terrible film. Waste of time.",
                    "Great acting and beautiful story.",
                    "Boring and predictable. Not recommended.",
                ] * 250,
                'label': [1, 0 , 1, 0] * 250
            })
            df.to_csv(csv_path, index=False)
        
    return csv_path

def build_vocab(texts, min_freq=2, max_size=10000):
    """Build vocabulary from texts"""
    from collections import Counter
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Count word frequencies
    counter = Counter()
    for text in texts:
        words = text.lower().split()
        counter.update(words)
    
    # Add most common words
    idx = 2
    for word, freq in counter.most_common(max_size - 2):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def get_dataloaders(batch_size=32, max_length=100, val_split=0.2):
    """Create train and validation dataloaders"""
    
    # Download/load dataset
    csv_path = download_imdb_dataset()
    df = pd.read_csv(csv_path)
    
    # For faster testing, use subset
    df = df.sample(n=min(25000, len(df)), random_state=42)  # Up to 5000 samples for quick training
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(df['text'].tolist())
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=val_split,
        random_state=42
    )
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, vocab, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, vocab = get_dataloaders(batch_size=4, max_length=20)
    
    # Test one batch
    texts, labels = next(iter(train_loader))
    print(f"Batch shape: {texts.shape}")
    print(f"Labels: {labels}")
    print(f"Vocab size: {len(vocab)}")        