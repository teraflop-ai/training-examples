import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "google-bert/bert-base-uncased"


class SingleInputMultiOutputBERT(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CHECKPOINT)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.bert.config.hidden_size, count)
            for name, count in classes.items()
        })

    def forward(self, **inputs):
        features = self.dropout(self.bert(**inputs).last_hidden_state[:, 0])
        return {name: head(features) for name, head in self.heads.items()}


texts = [
    "This phone has excellent battery life.",
    "These headphones broke after a week.",
]
labels = {
    "sentiment": torch.tensor([2, 0]),
    "category": torch.tensor([0, 1]),
}

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
inputs = tokenizer(
    texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
)
model = SingleInputMultiOutputBERT({"sentiment": 3, "category": 2})
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
optimizer.zero_grad()
logits = model(**inputs)
loss = sum(nn.functional.cross_entropy(logits[k], labels[k]) for k in labels)
loss.backward()
optimizer.step()

new_inputs = tokenizer(
    "These headphones sound excellent.",
    truncation=True, max_length=128, return_tensors="pt",
)
model.eval()
with torch.inference_mode():
    predictions = {
        name: scores.argmax(dim=-1).tolist()
        for name, scores in model(**new_inputs).items()
    }
print(predictions)