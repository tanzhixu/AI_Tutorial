import torch
from model import Model 
import json
import tiktoken

# Hyperparameters
batch_size = 4  # How many batches per training step
context_length = 16  # Length of the token chunk each batch
d_model = 64  # The size of our model token embeddings
num_blocks = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi-head attention
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 5000  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 准备训练数据
# https://hf-mirror.com/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true
with open('train_data.txt', 'r', encoding='utf-8') as f:
    alpaca = json.load(f)
    text = alpaca[1000:5001]
    
    
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(str(text))
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

total_tokens = encoding.encode_ordinary(str(text))
print(f"Total tokens: {total_tokens}")

# Split train and validation data
train_size = int(0.9 * len(tokenized_text))
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

model = Model()
model.load_state_dict(torch.load('model/model-scifi.pt'))
model.to(device)

# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model\\model-scifi-finetune.pt')