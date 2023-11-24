from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer


def BERT(num_classes: int):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )


def Tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")
