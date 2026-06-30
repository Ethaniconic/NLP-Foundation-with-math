# predict.py
import torch
from predictor import SentimentPredictor
from data_loader import get_dataloaders

def main():
    print("Loading model...")
    
    # Load vocab
    _, _, vocab = get_dataloaders(batch_size=1, max_length=10)
    
    # Create predictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = SentimentPredictor(
        model_path='checkpoints/best_model.pth',
        vocab=vocab,
        device=device
    )
    
    print("\n" + "="*60)
    print("MOODMETER - Movie Review Sentiment Analyzer")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        review = input("Enter a movie review: ").strip()
        
        if review.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not review:
            continue
        
        # Predict
        sentiment, confidence = predictor.predict(review)
        
        # Display result
        emoji = "😊" if sentiment == "Positive" else "😞"
        print(f"\n{emoji} Sentiment: {sentiment}")
        print(f"📊 Confidence: {confidence:.2%}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()