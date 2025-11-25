
import os
import sys
import torch
from transformers import AutoTokenizer

# Add current dir to path to find patched_tokenization_emu3
sys.path.append(os.getcwd())

try:
    from patched_tokenization_emu3 import Emu3Tokenizer
    print("Imported Emu3Tokenizer successfully.")
except ImportError as e:
    print(f"Could not import Emu3Tokenizer: {e}")
    sys.exit(1)

model_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu3.5-Image"
vocab_file = os.path.join(model_path, "emu3.tiktoken")

if os.path.exists(vocab_file):
    tokenizer = Emu3Tokenizer(vocab_file=vocab_file)
    print(f"Loaded tokenizer from {vocab_file}")
else:
    print("Vocab file not found.")
    sys.exit(1)

special_tokens = ["<|extra_203|>", "<|extra_100|>", "USER:", "ASSISTANT:"]

for t in special_tokens:
    ids = tokenizer.encode(t, add_special_tokens=False)
    print(f"Token '{t}' -> IDs: {ids}")

prompt = "<|extra_203|>You are a helpful assistant. USER: test ASSISTANT: <|extra_100|>"
ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"Full prompt IDs: {ids}")

# Check if extra_203 is a single token
id_203 = tokenizer.encode("<|extra_203|>", add_special_tokens=False)
if len(id_203) != 1:
    print("CRITICAL: <|extra_203|> is NOT a single token!")
else:
    print(f"OK: <|extra_203|> is token ID {id_203[0]}")
