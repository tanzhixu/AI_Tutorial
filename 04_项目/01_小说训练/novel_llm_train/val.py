import torch
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

batch_size = config.getint('Hyperparameters', 'batch_size')
context_length = config.getint('context_length', 'batch_size')
d_model = config.getint('max_iters', 'd_model')
num_blocks = config.getint('Hyperparameters', 'num_blocks')
num_heads = config.getint('Hyperparameters', 'num_heads')
learning_rate = config.getfloat('Hyperparameters', 'learning_rate')
dropout = config.getfloat('Hyperparameters', 'dropout')
max_iters = config.getint('Hyperparameters', 'max_iters')
eval_interval = config.getint('Hyperparameters', 'eval_interval')
eval_iters = config.getint('Hyperparameters', 'eval_iters')
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

with open('data/scifi.txt', 'r') as f:
    text = f.read()
    
vocab = sorted(list(set(text)))
vocab_size = max_token_value = len(vocab)
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for i,c in enumerate(vocab)}
encode = lambda x: [char2idx[c] for c in x]
decode = lambda idxs: ''.join([idx2char[i] for i in idxs])
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

model = Model(max_token_value=vocab_size).to(device)
model.load_state_dict(torch.load('model/model-ckpt.pt'))
model.eval()

start = '奥特曼出生在一个小村庄'
start_ids = encode(start)
x = (torch.tensor(torch.tensor(start_ids,dtype=torch.long,device=device)[None,...]))
y = model.generate(x, max_new_tokens=500)
print('-----------------')
print(decode(y[0].tolist()))
print('-----------------')