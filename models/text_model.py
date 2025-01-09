from transformers import AutoModelForSequenceClassification

def get_text_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=3  # Sentiment classes
    )
    return model
