from bert_score import score
from sentence_transformers import models
import torch.nn.functional as F

class Bertscore:

    def __init__(self, model_name="roberta-large"):
        self.model_name = model_name

    def download_model(self, model_name, model_path):
        model = models.Transformer("roberta-large")
        model.save(model_path)
        print(f"Model and tokenizer saved to {model_path}")

    def compute(self, candidates, references):
        P, R, F1 = score(candidates, references, model_type=self.model_name, lang="en", num_layers=17) 
        return F1.numpy()