import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "google-bert/bert-base-uncased"


class MultiInputOutputBERT(nn.Module):
    def __init__(self, fields, classes):
        super().__init__()
        self.fields = tuple(fields)
        self.bert = AutoModel.from_pretrained(CHECKPOINT)
        self.fusion = nn.Sequential(
            nn.Linear(len(self.fields) * self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(256, count) for name, count in classes.items()
        })

    def forward(self, inputs):
        vectors = [
            self.bert(**inputs[field]).last_hidden_state[:, 0]
            for field in self.fields
        ]
        features = self.fusion(torch.cat(vectors, dim=-1))
        return {name: head(features) for name, head in self.heads.items()}


texts = {
    "title": ["Great phone", "Poor headphones"],
    "description": ["Long battery life.", "Wireless headphones."],
    "review": ["Works perfectly.", "Broke after a week."],
}
labels = {
    "sentiment": torch.tensor([2, 0]),
    "category": torch.tensor([0, 1]),
}

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
inputs = {
    field: tokenizer(
        values, padding=True, truncation=True, max_length=128,
        return_tensors="pt",
    )
    for field, values in texts.items()
}
model = MultiInputOutputBERT(texts.keys(), {"sentiment": 3, "category": 5})
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = sum(nn.functional.cross_entropy(logits[k], labels[k]) for k in labels)
loss.backward()
optimizer.step()

model.eval()
with torch.inference_mode():
    predictions = {k: v.argmax(dim=-1).tolist() for k, v in model(inputs).items()}
print(predictions)