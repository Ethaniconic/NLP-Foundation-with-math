# src/trainer.py - UPDATED VERSION
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=0.001, patience=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience  # Early stopping patience
        
        self.criterion = nn.BCELoss()
        
        # CHANGE 3: Added weight_decay for regularization
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for texts, labels in pbar:
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(texts).squeeze()
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            predicted_labels = (predictions >= 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            
            for texts, labels in pbar:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(texts).squeeze()
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                predicted_labels = (predictions >= 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_path='checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_path, 'best_model.pth'))
                print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
            
            # EARLY STOPPING CHECK
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                print(f"  ⚠ No improvement in val loss ({self.epochs_without_improvement}/{self.patience})")
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n⚡ Early stopping triggered! Stopping at epoch {epoch+1}")
                    break
            
            print("-" * 60)
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.val_accuracies, label='Val Accuracy', marker='s', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        print("Training curves saved to 'training_curves.png'")
        plt.show()


if __name__ == "__main__":
    from model import SentimentLSTM
    from data_loader import get_dataloaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_loader, val_loader, vocab = get_dataloaders(batch_size=32, max_length=50)
    
    model = SentimentLSTM(vocab_size=len(vocab))
    
    trainer = Trainer(model, train_loader, val_loader, device=device)
    
    trainer.train(num_epochs=2)