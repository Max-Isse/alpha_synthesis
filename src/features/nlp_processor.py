from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBertSentimentEngine:
    def __init__(self):
        # Target local execution via GPU if available, else fallback to CPU thread pooling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ProsusAI/finbert"
        
        # Initialize tokenization and base weights safely
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval() # Explicitly lock evaluation mode to turn off dropout/batchnorm normalization

    def analyze_text_sentiment(self, text_chunks: List[str]) -> Dict[str, float]:
        """
        Tokenizes and infers financial sentiment scores over batched text blocks.
        Returns a dictionary containing mean probability mappings for positive, negative, and neutral tone.
        """
        if not text_chunks:
            return {"nlp_positive": 0.0, "nlp_negative": 0.0, "nlp_neutral": 1.0}

        # Tokenize text with uniform padding and truncation up to max model sequence length
        inputs = self.tokenizer(
            text_chunks, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad(): # Disable gradient tape tracking to conserve VRAM/RAM
            outputs = self.model(**inputs)
            # Apply standard softmax vector scaling to get relative probabilities
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # FinBERT labels: [0: Positive, 1: Negative, 2: Neutral]
        mean_scores = probabilities.mean(axis=0)
        
        return {
            "nlp_positive": float(mean_scores[0]),
            "nlp_negative": float(mean_scores[1]),
            "nlp_neutral": float(mean_scores[2])
        }

if __name__ == "__main__":
    # Test sample simulating a negative shift in management tone regarding debt covenant conditions
    mock_mda_snippet = [
        "Operating conditions have become highly challenging due to microeconomic pressures.",
        "We note increased liquidity constraints that may impair our ability to fund capital expenditures as expected."
    ]
    
    print("Initializing weights and loading FinBERT...")
    engine = FinBertSentimentEngine()
    metrics = engine.analyze_text_sentiment(mock_mda_snippet)
    
    print("\n--- Point-In-Time NLP Feature Extraction ---")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")