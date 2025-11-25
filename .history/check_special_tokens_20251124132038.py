
import os
import sys
import torch
from transformers import AutoTokenizer

# Add current dir to path to find patched_tokenization_emu3
sys.path.append(os.getcwd())

try:
    from patched_tokenization_emu3 import Emu3Tokenizer
except ImportError:
    print("Could not import Emu3Tokenizer.")
    sys.exit(1)

model_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu3.5-Image"
vocab_file = os.path.join(model_path, "emu3.tiktoken")

if os.path.exists(vocab_file):
    tokenizer = Emu3Tokenizer(vocab_file=vocab_file)
else:
    print("Vocab file not found.")
    sys.exit(1)

# Hardcoded IDs from logits_processor.py
HARDCODED_IDS = {
    "BOS": 151849,
    "EOS": 151850,
    "IMG": 151851,
    "BOI": 151852,
    "EOI": 151853,
    "EOL": 151846,
    "EOF": 151847,
    "BOV": 151854
}

print("Checking Tokenizer IDs vs Hardcoded IDs:")
for name, hard_id in HARDCODED_IDS.items():
    # We can't easily look up by ID in tiktoken without decoding, 
    # but we can check if the tokenizer has attributes or special tokens.
    # Emu3Tokenizer doesn't expose a direct name-to-id map for these specific visual tokens easily
    # unless they are in special_tokens.
    
    # Let's try to decode the hardcoded ID and see what it is.
    try:
        decoded = tokenizer.decode([hard_id])
        print(f"ID {hard_id} ({name}) decodes to: '{decoded}'")
    except Exception as e:
        print(f"ID {hard_id} ({name}) failed to decode: {e}")

# Check what the tokenizer thinks these special tokens are (if they exist in the class)
print("\nTokenizer Special Tokens:")
special_tokens_map = {
    "boi_token": getattr(tokenizer, "boi_token", "N/A"),
    "eoi_token": getattr(tokenizer, "eoi_token", "N/A"),
    "img_token": getattr(tokenizer, "img_token", "N/A"),
    "eol_token": getattr(tokenizer, "eol_token", "N/A"),
}

for name, token_str in special_tokens_map.items():
    if token_str != "N/A":
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        print(f"Attribute {name} ('{token_str}') encodes to: {ids}")
    else:
        print(f"Attribute {name} not found on tokenizer.")
