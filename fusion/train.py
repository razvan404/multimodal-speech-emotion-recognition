import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class FusionTrainer:
    _epochs_print = 20
    _num_epochs = 10

    @classmethod
    def train(cls, model: nn.Module, train_dataloader: DataLoader, num_classes: int):
        print("Loading the fusion configs...")
        optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        cross_entropy = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        history_loss = []
        history_acc = []

        print("Training the fusion model...")
        for epoch in range(cls._num_epochs):
            lr_scheduler.step()
            last_epochs_loss = 0.0
            last_epochs_acc = 0.0
            lost_steps = 0
            for train_step, batch in enumerate(train_dataloader, start=1):
                audio, text, emotion = batch
                if audio.shape[2] > 65:
                    emotion_one_hot = torch.tensor(
                        np.array(
                            [0.0 if emotion != i else 1.0 for i in range(num_classes)]
                        )
                    ).unsqueeze(0)
                    optimizer.zero_grad()
                    input_tokens = torch.tensor(
                        tokenizer.encode(text, add_special_tokens=True)
                    ).unsqueeze(0)
                    output = model(input_tokens, audio)
                    loss = cross_entropy(output, emotion_one_hot)
                    loss.backward()
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == emotion)

                    last_epochs_loss += loss.item()
                    last_epochs_acc += accuracy.item()
                    if (train_step - lost_steps) % cls._epochs_print == 0:
                        print(
                            f"EPOCH {epoch} \tSTEP {train_step} \tTRAINING LOSS {last_epochs_loss / cls._epochs_print}  "
                            f"\tTRAINING ACC {last_epochs_acc / cls._epochs_print}"
                        )
                        history_loss.append(last_epochs_loss)
                        history_acc.append(last_epochs_acc)

                        last_epochs_loss = 0.0
                        last_epochs_acc = 0.0
                else:
                    lost_steps += 1

        plt.title("Fusion Loss History")
        plt.plot(history_loss)
        plt.show()

        plt.title("Fusion Accuracy History")
        plt.plot(history_acc)
        plt.show()

        return tokenizer
