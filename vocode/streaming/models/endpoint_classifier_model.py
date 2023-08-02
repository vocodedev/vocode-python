import string
import torch.nn.functional as F
import time
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class EndpointClassifier:
    def __init__(self, checkpoint_path="Cyleux/checkpoint-42500"):
        self.device = torch.device("cuda")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            checkpoint_path
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.model.to(self.device)

    def classify_text(self, text, return_as_int=False):
        # Make the text lowercase
        text = text.lower()
        # Remove all punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Prepare the text data
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        # Move the inference to GPU if available
        inputs = inputs.to(self.device)
        # Run the text through the model
        outputs = self.model(**inputs)
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs.logits, dim=-1)
        # Get the predicted class
        _, predicted_class = torch.max(outputs.logits, dim=1)
        # Convert predicted class to boolean
        classification = bool(predicted_class.item())
        probability = probabilities[0][
            predicted_class
        ]  # this is the probability the sentence is complete
        if predicted_class == 0:
            probability = 1 - probability
        # if return_as_int:
        #     return probability
        if probability > 0.15:
            return True
        else:
            return False
