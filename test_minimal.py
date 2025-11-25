"""
Minimal test - does the model generate ANYTHING coherent?
Uses the exact same approach as your working ComfyUI node.
"""
import sys
import os
import torch
from PIL import Image
import numpy as np

# Add repo to path
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
sys.path.insert(0, emu_repo_path)
sys.path.insert(0, current_dir)

from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.vision_tokenizer import build_vision_tokenizer
from patched_tokenization_emu3 import Emu3Tokenizer
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

model_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu3.5-Image"
vq_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\vision_tokenizer"
device = "cuda:0"

print("Loading model...")
config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
model = Emu3ForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True,
)
model.eval()

# Fix missing generate method
if not hasattr(model, "generate"):
    print("Adding GenerationMixin...")
    cls = model.__class__
    if GenerationMixin not in cls.__bases__:
        cls.__bases__ = (GenerationMixin,) + cls.__bases__

print("Loading tokenizer...")
tokenizer = Emu3Tokenizer.from_pretrained(model_path)

# Fix missing generation_config (needs tokenizer loaded first)
if not hasattr(model, "generation_config") or model.generation_config is None:
    print("Setting default generation_config...")
    model.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("Loading VQ model...")
vq_model = build_vision_tokenizer("ibq", vq_path, device=device)

# Test parameters
prompt = "a red apple on a table"
H, W = 64, 64  # 1024x1024 pixels

# Format prompt
full_prompt = f"<|extra_203|>{prompt}<|extra_204|>{H}*{W}<|image|>"
print(f"\nPrompt: {full_prompt}")

input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
print(f"Input shape: {input_ids.shape}")

# Generate with MINIMAL config - no CFG, no logits processors
generation_config = GenerationConfig(
    max_new_tokens=H * (W + 1) + 50,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    do_sample=True,
    temperature=1.0,
    top_k=10240,
)

print("\nGenerating (no CFG, no logits processors)...")
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        generation_config=generation_config,
    )

# Extract generated tokens
generated = outputs[0, input_ids.shape[1]:].tolist()
print(f"\nGenerated {len(generated)} tokens")
print(f"First 20: {generated[:20]}")
print(f"Last 20: {generated[-20:]}")

# Check what stopped generation
EOS = 151850
if EOS in generated:
    eos_pos = generated.index(EOS)
    print(f"EOS token found at position {eos_pos}")
else:
    print("No EOS token found - hit max_new_tokens limit")

# Decode image
BOI = 151852
EOI = 151853
EOL = 151846
VISUAL_START = 151855

try:
    boi_idx = generated.index(BOI)
    eoi_idx = generated.index(EOI)
    print(f"Image tokens from {boi_idx} to {eoi_idx}")
    
    image_tokens = generated[boi_idx + 1:eoi_idx]
    
    # Extract visual indices
    rows = []
    current_row = []
    for tok in image_tokens:
        if tok == EOL:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif tok >= VISUAL_START:
            visual_idx = tok - VISUAL_START
            current_row.append(visual_idx)
    
    if current_row:
        rows.append(current_row)
    
    print(f"Decoded {len(rows)} rows")
    if rows:
        print(f"Row lengths: {[len(r) for r in rows[:5]]}...")
        print(f"First 10 indices of first row: {rows[0][:10]}")
        
        # Convert to tensor
        token_array = np.array(rows, dtype=np.int32)
        token_tensor = torch.from_numpy(token_array).unsqueeze(0).to(device)
        print(f"Token tensor shape: {token_tensor.shape}")
        
        # Decode with VQ
        with torch.no_grad():
            image_tensor = vq_model.decode_code(token_tensor)
        
        # Save
        img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        output_path = "test_minimal_nocfg.png"
        img.save(output_path)
        print(f"\n✅ Saved to: {output_path}")
        print("CHECK THIS IMAGE - is it coherent or garbage?")
        
except ValueError as e:
    print(f"❌ Failed to find image tokens: {e}")
except Exception as e:
    print(f"❌ Decoding failed: {e}")
    import traceback
    traceback.print_exc()
