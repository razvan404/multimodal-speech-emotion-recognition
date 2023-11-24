import torch
import torch.nn as nn
import transformers

from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix

from core.config import CONFIG
from text.bert import BERT, Tokenizer

NUM_EPOCHS = 4


class TextTrainer:
    @classmethod
    def train(
        cls,
        train_dataloader: DataLoader,
    ):
        classes = CONFIG.dataset_emotions()
        model = BERT(num_classes=len(classes))
        tokenizer = Tokenizer()
        optimizer = transformers.AdamW(model.parameters(), lr=1e-4, eps=1e-8)
        total_steps = len(train_dataloader) * train_dataloader.batch_size * NUM_EPOCHS
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        epochs_print = 10

        for epoch in range(NUM_EPOCHS):
            model.train()
            last_epochs_loss = 0
            last_epochs_acc = 0
            for train_step, batch in enumerate(train_dataloader, start=1):
                text, emotion = batch[1], batch[2]
                input_tokens = torch.tensor(
                    tokenizer.encode(text, add_special_tokens=True)
                ).unsqueeze(0)
                model.zero_grad()
                result = model(input_tokens, labels=emotion)
                loss, logits = result[0], result[1]
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                _, preds = torch.max(logits, 1)
                accuracy = torch.sum(preds == emotion)
                last_epochs_loss += loss.item()
                last_epochs_acc += accuracy.item()
                if train_step % epochs_print == 0:
                    print(
                        f"EPOCH {epoch} \tSTEP {train_step} \tTRAINING LOSS {last_epochs_loss / epochs_print}  "
                        f"\tTRAINING ACC {last_epochs_acc / epochs_print}"
                    )
                    last_epochs_loss = 0
                    last_epochs_acc = 0

        return model, tokenizer

    @classmethod
    def eval(cls, model: nn.Module, tokenizer, test_dataloader: DataLoader):
        y_actual = []
        y_pred = []
        model.eval()
        for batch in test_dataloader:
            text, emotion = batch[1], batch[2]
            input_tokens = torch.tensor(
                tokenizer.encode(text, add_special_tokens=True)
            ).unsqueeze(0)
            with torch.no_grad():
                result = model(input_tokens, labels=emotion)
                loss, logits = result[0], result[1]
                _, preds = torch.max(logits, 1)
                y_actual.append(emotion.numpy()[0])
                y_pred.append(preds.numpy()[0])
        print("Confusion matrix for test data:")
        print(confusion_matrix(y_actual, y_pred))
