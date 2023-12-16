import numpy as np
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
import logging
import tqdm

from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix

from vizualisers.plots import PlotVisualizer

logger = logging.getLogger(__name__)


class TextTrainer:
    def __init__(
        self,
        text_model: nn.Module,
        tokenizer: callable,
        num_epochs: int = 3,
        epochs_print: int = 20,
        learning_rate: float = 2e-5,
        loss: callable = nn.CrossEntropyLoss(),
    ):
        self._text_model = text_model
        self._tokenizer = tokenizer
        self._num_epochs = num_epochs
        self._epochs_print = epochs_print
        self._learning_rate = learning_rate
        self._optimizer = torch.optim.AdamW(
            self._text_model.parameters(), lr=self._learning_rate, eps=1e-8
        )
        self.loss = loss

    def train(
        self,
        train_dataloader: DataLoader,
    ):
        logger.info("Initializing the text configurations...")
        total_steps = (
            len(train_dataloader) * train_dataloader.batch_size * self._num_epochs
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history_loss = []
        history_acc = []

        logger.info("Training the text model...")
        self._text_model.train()
        for epoch in range(self._num_epochs):
            epoch_loss = 0
            epoch_acc = 0.0
            loader = tqdm.tqdm(
                enumerate(train_dataloader, start=1), desc=f"Epoch {epoch}"
            )
            for train_step, batch in loader:
                text, emotion = batch[1], batch[2][0]
                input_tokens = torch.tensor(
                    self._tokenizer.encode(
                        text, add_special_tokens=True, is_split_into_words=True
                    )
                ).unsqueeze(0)
                self._text_model.zero_grad()
                logits = self._text_model(input_tokens)[0]
                loss = self.loss(logits, emotion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._text_model.parameters(), 1.0)
                self._optimizer.step()
                scheduler.step()
                _, preds = torch.max(logits, 1)
                accuracy = torch.sum(preds == emotion)

                loss = loss.item()
                accuracy = accuracy.item()
                history_loss.append(loss)
                history_acc.append(accuracy)
                epoch_loss += loss
                epoch_acc += accuracy
                loader.set_postfix(
                    loss=epoch_loss / train_step, accuracy=epoch_acc / train_step
                )

        self.plot_histories(history_loss, history_acc)

    @classmethod
    def plot_histories(
        cls, history_losses: list[float], history_accuracies: list[float]
    ):
        PlotVisualizer.plot_many(
            (1, 2),
            lambda: PlotVisualizer.plot_history(history_losses, "Text Loss History"),
            lambda: PlotVisualizer.plot_history(
                history_accuracies, "Text Accuracy History"
            ),
        )
        plt.show()

    def eval(self, test_dataloader: DataLoader):
        y_actual = []
        y_pred = []
        self._text_model.eval()
        for batch in test_dataloader:
            text, emotion = batch[1], batch[2]
            input_tokens = torch.tensor(self._tokenizer.encode(text)).unsqueeze(0)

            with torch.no_grad():
                result = self._text_model(input_tokens, labels=emotion)
                loss, logits = result[0], result[1]
                _, preds = torch.max(logits, 1)
                y_actual.append(emotion.numpy()[0])
                y_pred.append(preds.numpy()[0])

        conf_matrix = confusion_matrix(y_actual, y_pred)
        print("Confusion matrix for test data:")
        print(conf_matrix)
        print("Accuracy:", np.sum(np.diag(conf_matrix)) / np.sum(np.diag()))
