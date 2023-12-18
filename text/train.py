import numpy as np
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
import logging
import tqdm

from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from vizualisers.plots import PlotVisualizer

logger = logging.getLogger(__name__)


class TextTrainer:
    def __init__(
        self,
        text_model: nn.Module,
        num_epochs: int = 3,
        epochs_print: int = 20,
        learning_rate: float = 2e-5,
        loss: callable = nn.CrossEntropyLoss(),
    ):
        self._text_model = text_model
        self._num_epochs = num_epochs
        self._epochs_print = epochs_print
        self._learning_rate = learning_rate
        self._optimizer = torch.optim.AdamW(
            self._text_model.parameters(), lr=self._learning_rate, eps=1e-8
        )
        self.loss = loss

    def train(self, train_dataloader: DataLoader):
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
        for epoch in range(1, self._num_epochs + 1):
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for batch in loader:
                # Forward
                text, emotion = batch[1], batch[2]
                logits = self._text_model(text)[0]
                loss = self.loss(logits, emotion)

                # Backward
                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._text_model.parameters(), 1.0)
                self._optimizer.step()
                scheduler.step()

                # Perform the accuracy
                _, preds = torch.max(logits, dim=1)
                accuracy = torch.mean((preds == emotion).float())

                # Register the metrics
                loss = loss.item()
                accuracy = accuracy.item()
                history_loss.append(loss)
                history_acc.append(accuracy)
                loader.set_postfix(loss=loss, accuracy=accuracy)

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

    def eval(self, test_dataloader: DataLoader, labels: list[str] = None):
        y_actual = []
        y_pred = []
        self._text_model.eval()
        with torch.no_grad():
            loader = tqdm.tqdm(test_dataloader, "Evaluating the model")
            for batch in loader:
                text, emotion = batch[1], batch[2]
                logits = self._text_model(text)[0]
                _, preds = torch.max(logits, 1)
                y_actual += emotion.numpy().tolist()
                y_pred += preds.numpy().tolist()

        print(len(y_actual), len(y_pred))
        possible_values = sorted({*y_actual, *y_pred})
        possible_labels = [
            label for i, label in enumerate(labels) if i in possible_values
        ]
        conf_matrix = confusion_matrix(y_actual, y_pred)
        print("Accuracy:", np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
        PlotVisualizer.plot_confusion_matrix(conf_matrix, possible_labels)
