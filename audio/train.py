import torch
import torch.nn as nn
import numpy as np
import transformers
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader


class AudioTrainer:
    _epochs_print = 20
    _num_epochs = 2

    @classmethod
    def train(cls, model: nn.Module, train_dataloader: DataLoader, num_classes: int):
        print("Loading the audio configuration...")
        optimizer = transformers.AdamW(model.parameters(), lr=2e-4, eps=1e-8)
        total_steps = (
            len(train_dataloader) * train_dataloader.batch_size * cls._num_epochs
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        cross_entropy = nn.CrossEntropyLoss()

        history_loss = []
        history_acc = []

        print("Training the audio model...")
        for epoch in range(cls._num_epochs):
            last_epochs_loss = 0.0
            last_epochs_acc = 0.0
            for train_step, batch in enumerate(train_dataloader, start=1):
                audio, _, emotion = batch
                if audio.shape[2] > 65:
                    emotion_one_hot = torch.tensor(
                        np.array(
                            [0.0 if emotion != i else 1.0 for i in range(num_classes)]
                        )
                    ).unsqueeze(0)
                    model.zero_grad()
                    logits = model(audio)
                    loss = cross_entropy(logits, emotion_one_hot)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    _, preds = torch.max(logits, 1)
                    accuracy = torch.sum(preds == emotion)
                    history_loss.append(loss.item())
                    history_acc.append(accuracy.item())
                    last_epochs_loss += loss.item()
                    last_epochs_acc += accuracy.item()
                    if train_step % cls._epochs_print == 0:
                        print(
                            f"EPOCH {epoch} \tSTEP {train_step} \tTRAINING LOSS {last_epochs_loss / cls._epochs_print}  "
                            f"\tTRAINING ACC {last_epochs_acc / cls._epochs_print}"
                        )
                        last_epochs_loss = 0.0
                        last_epochs_acc = 0.0

        plt.plot(history_loss)
        plt.title("Loss History")
        plt.show()

        print(history_loss)
        print(history_acc)

        plt.plot(history_acc)
        plt.title("Accuracy history")
        plt.show()

        return model

    @classmethod
    def eval(cls, model, test_dataloader: DataLoader):
        y_actual = []
        y_pred = []
        model.eval()
        for batch in test_dataloader:
            pass
