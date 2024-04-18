import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import Model 

import configparser

def read_config_file(file_path):
    # 创建 ConfigParser 对象
    config = configparser.ConfigParser()
    # 读取配置文件
    config.read(file_path)
    return config

config_file_path = 'hyper-parameters.ini'
config = read_config_file(config_file_path)

# Hyperparameters
batch_size = config.getint('Hyperparameters', 'batch_size')
context_length = config.getint('context_length', 'context_length')
max_iters = config.getint('max_iters', 'max_iters')
learning_rate = config.getfloat('Hyperparameters', 'learning_rate')
eval_interval = config.getint('Hyperparameters', 'eval_interval')
eval_iters = config.getfloat('Hyperparameters', 'eval_iters')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 准备训练数据
with open('data/scifi.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
vocab = sorted(list(set(text)))
vocab_size = max_token_value = len(vocab)
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for i,c in enumerate(vocab)}
encode = lambda x: [char2idx[c] for c in x]
decode = lambda idxs: ''.join([idx2char[i] for i in idxs])
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# Split train and validation set
train_size = int(0.8 * len(tokenized_text))
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# 初始化模型
model = Model(max_token_value=vocab_size).to(device)

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
torch.save(model.state_dict(), 'model/model-ckpt.pt')