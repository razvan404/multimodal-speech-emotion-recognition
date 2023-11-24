import torch
from torch import nn
from transformers import BertTokenizer


class CombinedAudioTextModel(nn.Module):
    def _hook_text(self, module, inputs, outputs):
        self._outputs_text.clear()
        self._outputs_text.append(outputs)
        return None

    def _hook_audio(self, module, inputs, outputs):
        self._outputs_audio.clear()
        self._outputs_audio.append(outputs)
        return None

    def __init__(self, num_classes: int):
        self._outputs_text = []
        self._outputs_audio = []
        super(CombinedAudioTextModel, self).__init__()
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.text_model = torch.load("saved_models/text_model.pt")
        self.audio_model = torch.load("saved_models/audio_model.pt")

        self.text_model.bert.pooler.register_forward_hook(self._hook_text)
        self.audio_model.features.register_forward_hook(self._hook_audio)

        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.audio_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1024, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, audio):
        self.text_model(text)
        self.audio_model(audio)
        audio_embed = self._outputs_audio[0]
        text_embed = self._outputs_text[0]
        audio_embed = torch.flatten(
            audio_embed, start_dim=2
        )  # a1,a2,a3......al{a of dim c}
        audio_embed = torch.sum(audio_embed, dim=2)
        concat_embded = torch.cat((text_embed, audio_embed), 1)
        x = self.dropout(concat_embded)
        x = self.linear(x)
        x = self.softmax(x)
        return x
