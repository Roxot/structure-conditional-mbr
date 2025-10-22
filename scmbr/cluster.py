import numpy as np

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, SentenceTransformerModelCardData
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import BatchSamplers

from smbr.datasets import construct_triplet_dataset

class ClusteringModel:

    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def train(self, data, output_dir, run_name=None ,lr=5e-5, nepochs=5):
        train_dataset = construct_triplet_dataset(data["train"])
        eval_dataset = construct_triplet_dataset(data["validation"])
        batch_size = 32 
        gradient_accumulation_steps = 4
        evals_per_epoch = 10
        loss = TripletLoss(model=self.model)
        eval_interval = max(1, len(train_dataset) // (evals_per_epoch * batch_size))
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,

            num_train_epochs=nepochs,
            per_device_train_batch_size=batch_size // gradient_accumulation_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            fp16=False,
            bf16=True, 
            batch_sampler=BatchSamplers.NO_DUPLICATES, 
            learning_rate=lr,

            eval_strategy="steps",
            eval_steps=eval_interval,
            save_strategy="steps",
            save_steps=eval_interval,
            save_total_limit=1,
            load_best_model_at_end=True,

            logging_steps=50,
            run_name=run_name
        )
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        print(trainer.evaluate())
        trainer.train()
        trainer.save_model(str(output_dir))
        self.model = SentenceTransformer(str(output_dir))

    def __forward__(self):
        return self.cluster()