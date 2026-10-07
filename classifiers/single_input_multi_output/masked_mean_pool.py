import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "jhu-clsp/ettin-encoder-150m"


class SingleInputMultiOutput(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            CHECKPOINT, attn_implementation="sdpa"
        )
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.encoder.config.hidden_size, count)
            for name, count in classes.items()
        })

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        features = self.dropout(pooled)
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


def encode(texts):
    return tokenizer(
        texts, padding=True, truncation=True, max_length=128,
        return_token_type_ids=False, return_tensors="pt",
    )


model = SingleInputMultiOutput({"sentiment": 3, "category": 2})
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
inputs = encode(texts)

model.train()
optimizer.zero_grad()
logits = model(**inputs)
loss = sum(nn.functional.cross_entropy(logits[k], labels[k]) for k in labels)
loss.backward()
optimizer.step()

model.eval()
with torch.inference_mode():
    logits = model(**encode(["These headphones sound excellent."]))
    predictions = {
        name: scores.argmax(dim=-1).tolist()
        for name, scores in logits.items()
    }
print(predictions)