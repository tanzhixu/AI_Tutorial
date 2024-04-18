import torch
from model import Model

model = Model(max_token_value=8193)
state_dict = torch.load('model/model-ckpt.pt')

model.load_state_dict(state_dict=state_dict)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'模型参数为： {total_params:,}')