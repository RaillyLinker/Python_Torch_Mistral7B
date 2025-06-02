import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from mistral import Mistral7B  # 사용자 정의 Mistral7B 클래스

# 1. 하이퍼파라미터 설정
BATCH_SIZE = 2
SEQ_LEN = 512
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_MODEL = "gpt2"  # 또는 "mistralai/Mistral-7B-v0.1"

# 2. 토크나이저 준비
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # GPT계열은 eos를 pad로 사용해도 무방

# 3. 데이터셋 로드 및 토크나이즈
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")


def tokenize(example):
    return tokenizer(example["text"], return_attention_mask=False)


tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=["text"])

# 4. 전체 토큰을 이어붙여서 SEQ_LEN 단위로 슬라이싱
class TextDataset(Dataset):
    def __init__(self, tokenized_data, seq_len):
        all_tokens = []
        for item in tokenized_data["input_ids"]:
            all_tokens.extend(item)
        total_len = len(all_tokens)
        self.seq_data = [
            torch.tensor(all_tokens[i:i + seq_len])
            for i in range(0, total_len - seq_len, seq_len)
        ]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        x = self.seq_data[idx]
        return x, x.clone()  # Input과 Target 동일

train_dataset = TextDataset(tokenized_datasets["train"], SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. Mistral7B 모델 초기화 (사용자 정의 클래스)
model = Mistral7B(
    vocab_size=tokenizer.vocab_size,
    dim=512,
    n_layers=4,
    num_heads=8,
    num_kv_heads=2,
    hidden_dim=2048,
    max_len=SEQ_LEN,
    window_size=SEQ_LEN,
).to(DEVICE)

# 6. 옵티마이저 및 손실 함수
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()  # pad_token_id를 사용하지 않으므로 ignore_index 생략

# 7. 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        x, y = [b.to(DEVICE) for b in batch]  # x, y: (B, T)

        optimizer.zero_grad()
        logits = model(x)  # (B, T, vocab_size)

        # Shift: logits[:, :-1, :] vs y[:, 1:]
        loss = loss_fn(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            y[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
    torch.save(model, "mistral7b_toy.pth")
