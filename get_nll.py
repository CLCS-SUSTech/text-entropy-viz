# %%
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
print('torch version:', torch.__version__)

# %%
# 加载 model 和 tokenizer
model_path = '/Users/xy/models/llama3-8b-base'  # 请根据实际情况修改

# 如果模型路径不存在，可以尝试其他路径
if not os.path.exists(model_path):
    print(f"模型路径不存在: {model_path}")
    print("请修改model_path为正确的模型路径")
    # 可以尝试其他常见路径
    possible_paths = [
        '/data1/model/llama3-8b-base',
        '/home/xy/models/llama3-8b-base',
        './models/llama3-8b-base'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"使用模型路径: {model_path}")
            break
    else:
        print("未找到可用的模型路径，请手动设置")
        exit(1)

assert os.path.exists(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 使用CPU进行计算（如果没有GPU）
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map='auto', 
                                             trust_remote_code=True).eval()


# %%
# Independent function for computing NLLs
def text_to_nlls(text, tokenizer, model):
    device = model.device
    ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(device)

    # Forward
    try:
        output = model(ids)
    except Exception:
        raise
    
    logits = output.logits.to(device)
    logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
    shift_logits = logits[:, :, :-1]
    shift_targets = ids[:, 1:]

    # NLL
    loss_fn = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)
    try:
        nlls = loss_fn(log_softmax(shift_logits), shift_targets)
        nlls = nlls.squeeze(0)
    except Exception:
        raise

    return nlls.detach().cpu().numpy()


# %%
# args
if len(sys.argv) != 3:
    print('Usage: python get_nll.py <text_file> <output_file>')
    exit(1)

text_file = sys.argv[1]
output_file = sys.argv[2]

# Read text file
with open(text_file, 'r') as f:
    text = f.read()

# Compute NLLs
nlls = text_to_nlls(text, tokenizer, model)

# Save NLLs
with open(output_file, 'w') as f:
    nlls_list = nlls.tolist()
    nlls_str = ' '.join([f'{nll:.4f}' for nll in nlls_list])
    f.write(nlls_str)
