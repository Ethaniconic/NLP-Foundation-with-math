# train.py - UPDATED VERSION
import torch
from model import SentimentLSTM
from data_loader import get_dataloaders
from trainer import Trainer

def main():
    # CHANGE 1 & 5: More data, fewer epochs
    BATCH_SIZE = 32
    MAX_LENGTH = 100
    NUM_EPOCHS = 15  # Reduced from 10
    LEARNING_RATE = 0.001
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading data...")
    train_loader, val_loader, vocab = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    print("\nInitializing model...")
    # CHANGE 2: Higher dropout (0.5)
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.5  # Increased from 0.3
    )
    
    # CHANGE 4: Early stopping with patience=3
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LEARNING_RATE,
        patience=3  # Stop if no improvement for 3 epochs
    )
    
    print("\n" + "="*60)
    trainer.train(num_epochs=NUM_EPOCHS, save_path='checkpoints')
    print("="*60)


if __name__ == "__main__":
    main()