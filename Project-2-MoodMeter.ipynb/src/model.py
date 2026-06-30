# src/model.py - UPDATED VERSION
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.5):
        # CHANGE 2: Default dropout increased from 0.3 to 0.5
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # CHANGE 2: Higher dropout
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = lstm_out[:, -1, :]
        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)
        output = self.sigmoid(output)
        
        return output

if __name__ == "__main__":
    vocab_size = 10000
    batch_size = 32
    seq_len = 100
    
    model = SentimentLSTM(vocab_size=vocab_size)
    
    dummy_text = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = model(dummy_text)
    
    print(f"Input shape: {dummy_text.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[:5].squeeze()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")