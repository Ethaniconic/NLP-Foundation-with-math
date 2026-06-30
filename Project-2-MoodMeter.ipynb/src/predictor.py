import torch
import re
from model import SentimentLSTM

class SentimentPredictor:
    def __init__(self, model_path, vocab, device='cpu', max_length=100):
        """
        Args:
            model_path: Path to saved model checkpoint
            vocab: Vocabulary dictionary (word -> index)
            device: 'cpu' or 'cuda'
            max_length: Maximum sequence length
        """
        self.device = device
        self.vocab = vocab
        self.max_length = max_length
        
        # Load model
        self.model = SentimentLSTM(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded from {model_path}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Split into words
        words = text.split()
        
        # Convert to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Padding or truncating
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor([indices], dtype=torch.long)
    
    def predict(self, text):
        """
        Predict sentiment of a single review
        
        Returns:
            sentiment: 'Positive' or 'Negative'
            confidence: Confidence score (0-1)
        """
        # Preprocess
        input_tensor = self.preprocess_text(text)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            confidence = output.item()
        
        # Interpret
        if confidence >= 0.5:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
            confidence = 1 - confidence  # Flip for negative
        
        return sentiment, confidence
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple reviews"""
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        return results


if __name__ == "__main__":
    # Test predictor
    from data_loader import get_dataloaders
    
    # Load vocab
    _, _, vocab = get_dataloaders(batch_size=1, max_length=10)
    
    # Create predictor
    predictor = SentimentPredictor(
        model_path='checkpoints/best_model.pth',
        vocab=vocab,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test reviews
    test_reviews = [
        "This movie was absolutely amazing! Great acting and story.",
        "Terrible film. Complete waste of time. Boring.",
        "I loved every moment of it. Highly recommended!",
        "Worst movie I have ever seen. Awful.",
        "The plot was predictable but the acting was good.",
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT PREDICTIONS")
    print("="*60)
    
    for review in test_reviews:
        sentiment, confidence = predictor.predict(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")