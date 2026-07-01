import pytest
from src.features.nlp_processor import FinBertSentimentEngine

@pytest.fixture(scope="module")
def sentiment_engine():
    """Fixture to initialize FinBERT once for the entire test module session."""
    return FinBertSentimentEngine()

def test_sentiment_bounds_and_structure(sentiment_engine):
    """Ensures the NLP engine outputs valid, bounded probability structures."""
    sample_text = ["Revenue increased by 20% year-over-year."]
    metrics = sentiment_engine.analyze_text_sentiment(sample_text)
    
    # Assert all keys exist
    for key in ["nlp_positive", "nlp_negative", "nlp_neutral"]:
        assert key in metrics
        # Assert probabilities are bounded between 0 and 1
        assert 0.0 <= metrics[key] <= 1.0

    # Assert that probabilities roughly sum to 1 (allowing for floating-point rounding)
    total_prob = metrics["nlp_positive"] + metrics["nlp_negative"] + metrics["nlp_neutral"]
    assert pytest.approx(total_prob, abs=1e-4) == 1.0

def test_directional_sentiment_accuracy(sentiment_engine):
    """Ensures that distinct negative management narrative shifts register accurately."""
    negative_text = ["We are facing severe regulatory headwinds and potential bankruptcy."]
    positive_text = ["We achieved record-breaking profits and scaled our core margins efficiently."]
    
    neg_metrics = sentiment_engine.analyze_text_sentiment(negative_text)
    pos_metrics = sentiment_engine.analyze_text_sentiment(positive_text)
    
    # Directional sanity checks
    assert neg_metrics["nlp_negative"] > pos_metrics["nlp_negative"], "Negative text should score higher negative tone."
    assert pos_metrics["nlp_positive"] > neg_metrics["nlp_positive"], "Positive text should score higher positive tone."