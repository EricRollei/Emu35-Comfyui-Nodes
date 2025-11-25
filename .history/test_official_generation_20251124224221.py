"""
Test script using the EXACT official Emu3.5 generation approach
to verify if the model weights and generation code are working correctly.
"""
import sys
import os
import torch

# Add repo to path
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
sys.path.insert(0, emu_repo_path)
sys.path.insert(0, current_dir)

from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.vision_tokenizer import build_vision_tokenizer
from src.utils.generation_utils import build_logits_processor, non_streaming_generate
from patched_tokenization_emu3 import Emu3Tokenizer

# Configuration matching official config.py
class TestConfig:
    def __init__(self, cfg_scale=1.0, h=64, w=64):
        self.classifier_free_guidance = cfg_scale
        self.image_area = h * w * 16 * 16  # pixel area
        self.h = h
        self.w = w
        self.target_height = h
        self.target_width = w
        self.image_cfg_scale = 1.0
        self.unconditional_type = "no_text"
        
        self.special_token_ids = {
            "PAD": 151643,
            "EOS": 151645,
            "BOS": 151849,
        }
        
        self.sampling_params = {
            "use_cache": True,
            "text_top_k": 1024,
            "text_top_p": 0.9,
            "text_temperature": 1.0,
            "image_top_k": 10240,
            "image_top_p": 1.0,
            "image_temperature": 1.0,
            "top_k": 131072,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams_per_group": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "max_new_tokens": h * (w + 1) + 50,
            "guidance_scale": cfg_scale,
            "use_differential_sampling": True,
            "do_sample": True,
            "num_beams": 1,
        }

def main():
    # Use the ComfyUI models path
    model_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu3.5-Image"
    vq_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\vision_tokenizer"
    device = "cuda:0"
    
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"VQ path: {vq_path}")
    print(f"VQ exists: {os.path.exists(vq_path)}")
    
    print("\nLoading config...")
    config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading model...")
    model = Emu3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = Emu3Tokenizer.from_pretrained(model_path)
    
    print("Loading VQ model...")
    vq_model = build_vision_tokenizer(vq_path).to(device).eval()
    
    # Test with CFG 1.0 (no guidance)
    cfg_scale = 1.0
    prompt = "a red apple on a table"
    H, W = 64, 64  # latent dimensions for 1024x1024
    
    print(f"\nTest: CFG {cfg_scale}, prompt: '{prompt}'")
    print(f"Target: {H}*{W} latents = {H*16}x{W*16} pixels")
    
    # Format prompt EXACTLY like official code
    full_prompt = f"<|extra_203|>{prompt}<|extra_204|>{H}*{W}<|image|>"
    full_negative = f"<|extra_203|><|extra_204|>{H}*{W}<|image|>"
    
    print(f"\nPositive prompt: {full_prompt}")
    print(f"Negative prompt: {full_negative}")
    
    # Tokenize
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    unconditional_ids = tokenizer.encode(full_negative, return_tensors="pt", add_special_tokens=False).to(device)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Unconditional shape: {unconditional_ids.shape}")
    
    # Build config
    cfg = TestConfig(cfg_scale=cfg_scale, h=H, w=W)
    
    # Generate using official function
    print("\nGenerating with official non_streaming_generate()...")
    
    try:
        with torch.no_grad():
            token_ids = non_streaming_generate(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                unconditional_ids=unconditional_ids,
                force_same_image_size=True,
            )
        
        print(f"Generated {token_ids.shape[1]} tokens")
        
        # Extract visual tokens
        generated = token_ids[0, input_ids.shape[1]:].tolist()
        print(f"\nFirst 20 generated tokens: {generated[:20]}")
        
        # Decode image
        BOI = 151852
        EOI = 151853
        EOL = 151846
        VISUAL_START = 151855
        
        try:
            boi_idx = generated.index(BOI)
            eoi_idx = generated.index(EOI)
            print(f"Found image tokens from {boi_idx} to {eoi_idx}")
            
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
            
            print(f"\nImage structure: {len(rows)} rows")
            if rows:
                print(f"First row length: {len(rows[0])}")
                print(f"First 10 indices: {rows[0][:10]}")
                
                # Decode with VQ model
                import numpy as np
                token_array = np.array(rows, dtype=np.int32)
                token_tensor = torch.from_numpy(token_array).unsqueeze(0).to(device)
                
                print(f"Token tensor shape: {token_tensor.shape}")
                
                with torch.no_grad():
                    image_tensor = vq_model.decode_code(token_tensor)
                
                # Save image
                from PIL import Image
                img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
                img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                
                output_path = "test_official_cfg1.png"
                img.save(output_path)
                print(f"\nSaved image to: {output_path}")
                print("Please check if this image is coherent or garbage!")
                
        except ValueError as e:
            print(f"Error finding image tokens: {e}")
            
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
