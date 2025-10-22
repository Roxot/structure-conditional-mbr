from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import torch

class BLEURT:

    def __init__(self, model_name="lucadiliello/BLEURT-20-D6"):
        self.tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D6")
        self.model = BleurtForSequenceClassification.from_pretrained(model_name)

    def compute(self, candidates, references):
        inputs = self.tokenizer(
            references, candidates, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = outputs.logits.squeeze(-1).cpu().numpy()
        return scores