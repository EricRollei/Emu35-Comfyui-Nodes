"""
Test with logits processor but NO CFG - to constrain tokens to valid image structure
"""
import sys
import os
import torch
from PIL import Image
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
sys.path.insert(0, emu_repo_path)
sys.path.insert(0, current_dir)

from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.vision_tokenizer import build_vision_tokenizer
from src.utils.generation_utils import build_logits_processor
from patched_tokenization_emu3 import Emu3Tokenizer
from transformers import GenerationConfig, LogitsProcessorList
from transformers.generation.utils import GenerationMixin

model_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu35-base"
vq_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\vision_tokenizer"
device = "cuda:0"

print(f"Testing with: {model_path}")

print("Loading components...")
config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
model = Emu3ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

if not hasattr(model, "generate"):
    cls = model.__class__
    if GenerationMixin not in cls.__bases__:
        cls.__bases__ = (GenerationMixin,) + cls.__bases__

tokenizer = Emu3Tokenizer.from_pretrained(model_path)

if not hasattr(model, "generation_config") or model.generation_config is None:
    model.generation_config = GenerationConfig(pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

vq_model = build_vision_tokenizer("ibq", vq_path, device=device)

# Test
prompt = "a red apple"
H, W = 64, 64

full_prompt = f"<|extra_203|>{prompt}<|extra_204|>{H}*{W}<|image|>"
input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)

# Empty unconditional for logits processor (but guidance scale = 1.0 = no CFG)
unconditional_prompt = f"<|extra_203|><|extra_204|>{H}*{W}<|image|>"
unconditional_ids = tokenizer.encode(unconditional_prompt, return_tensors="pt", add_special_tokens=False).to(device)

# Config for logits processor
class DummyConfig:
    def __init__(self):
        self.classifier_free_guidance = 1.0  # NO CFG
        self.image_area = H * W * 16 * 16
        self.h = H
        self.w = W
        self.target_height = H
        self.target_width = W
        self.image_cfg_scale = 1.0
        self.unconditional_type = "no_text"
        self.special_token_ids = {"PAD": tokenizer.pad_token_id, "EOS": tokenizer.eos_token_id, "BOS": 151849}
        self.sampling_params = {
            "use_cache": True, "max_new_tokens": H * (W + 1) + 50,
            "top_k": 10240, "top_p": 1.0, "temperature": 1.0, "do_sample": True,
            "use_differential_sampling": True,
            "guidance_scale": 1.0,
        }

cfg = DummyConfig()
logits_processor = LogitsProcessorList()

try:
    processor = build_logits_processor(cfg=cfg, unconditional_ids=unconditional_ids, model=model, tokenizer=tokenizer, force_same_image_size=True)
    logits_processor.append(processor)
    print("✅ Logits processor built successfully")
except Exception as e:
    print(f"❌ Failed to build logits processor: {e}")
    print("Continuing without it...")

generation_config = GenerationConfig(
    max_new_tokens=H * (W + 1) + 50,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    do_sample=True,
    temperature=1.0,
    top_k=10240,
)

print(f"\nGenerating with logits processor (CFG=1.0)...")
with torch.no_grad():
    outputs = model.generate(input_ids, generation_config=generation_config, logits_processor=logits_processor)

generated = outputs[0, input_ids.shape[1]:].tolist()
print(f"Generated {len(generated)} tokens")

# Decode
BOI, EOI, EOL, VISUAL_START = 151852, 151853, 151846, 151855

try:
    if BOI not in generated:
        print(f"❌ No BOI token found in output")
        print(f"Tokens present: {set(generated)}")
        exit(1)
    if EOI not in generated:
        print(f"❌ No EOI token found in output - generation didn't complete properly")
        print(f"First 50 tokens: {generated[:50]}")
        print(f"Last 50 tokens: {generated[-50:]}")
        exit(1)
    
    boi_idx = generated.index(BOI)
    eoi_idx = generated.index(EOI)
    image_tokens = generated[boi_idx + 1:eoi_idx]
    
    rows = []
    current_row = []
    for tok in image_tokens:
        if tok == EOL:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif tok >= VISUAL_START:
            current_row.append(tok - VISUAL_START)
    
    if current_row:
        rows.append(current_row)
    
    print(f"Decoded {len(rows)} rows")
    if rows:
        row_lens = [len(r) for r in rows]
        print(f"Row lengths - min: {min(row_lens)}, max: {max(row_lens)}, first 10: {row_lens[:10]}")
        
        if len(set(row_lens)) == 1:  # All same length
            token_array = np.array(rows, dtype=np.int32)
            token_tensor = torch.from_numpy(token_array).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_tensor = vq_model.decode_code(token_tensor)
            
            img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            output_path = "test_with_processor_nocfg.png"
            img.save(output_path)
            print(f"\n✅ SUCCESS! Saved to: {output_path}")
            print("CHECK THIS IMAGE!")
        else:
            print(f"❌ Rows have inconsistent lengths - can't form grid")
            
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
