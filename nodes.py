import os
import sys
import torch
import folder_paths
import comfy.model_management
import comfy.utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer
import torch.nn.functional as F

class ComfyStreamer(BaseStreamer):
    def __init__(self, total_steps):
        self.pbar = comfy.utils.ProgressBar(total_steps)
        self.counter = 0
        self.total_steps = total_steps

    def put(self, value):
        self.counter += 1
        self.pbar.update(1)
        return value

    def end(self):
        pass

# Add the Emu3.5 repo to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")

# Register emu35 model folder
try:
    folder_paths.add_model_folder_path("emu35", os.path.join(folder_paths.models_dir, "emu35"))
except:
    pass

if os.path.exists(emu_repo_path):
    sys.path.append(emu_repo_path)
    try:
        from src.emu3p5 import Emu3ForCausalLM, Emu3Config
        from src.vision_tokenizer import build_vision_tokenizer
        from src.utils.generation_utils import build_logits_processor, decode_image as official_decode_image, multimodal_decode
        # Note: Tokenizer is loaded via AutoTokenizer.from_pretrained with trust_remote_code=True
        # This loads tokenization_emu3.py from the model directory (we copy our patched version there)
    except ImportError as e:
        print(f"Error importing Emu3.5 modules: {e}")
        print("Please ensure the Emu3.5 repository is cloned into 'Emu3_5_repo' inside this node's directory.")

# Fallback local implementation of decode_image only if official import fails
import re
from typing import List
from PIL import Image
import numpy as np

# Constants for token IDs
BOI_TOKEN_ID = 151852  # <|image start|>
EOI_TOKEN_ID = 151853  # <|image end|>
IMG_TOKEN_ID = 151851  # <|image|>

def decode_image(image_string, tokenizer, vision_tokenizer):
    """
    Official Emu3.5 decode_image - exact match from generation_utils.py
    CRITICAL: No width padding, hardcoded embed_dim=256
    """
    import re
    from typing import List
    
    image: List[List[int]] = []
    image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            image.append(row_token)
    try:
        image = torch.tensor(
            image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
        )
        h, w = image.shape
        # CRITICAL: Official code ALWAYS uses 256 for embed_dim
        image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, 256)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(
            ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        return image
    except Exception as ex:
        print(f"decode_image failed: {ex}")
        import traceback
        traceback.print_exc()
        return None
# End of local decode_image

if not os.path.exists(emu_repo_path):
    print(f"Emu3.5 repository not found at {emu_repo_path}")
    print("Please clone https://github.com/baaivision/Emu3.5 into 'Emu3_5_repo' inside this node's directory.")

def get_emu_subfolders():
    # Helper to list subdirectories in the emu35 model folder
    # This keeps the dropdown clean (showing folders instead of 100s of shard files)
    model_paths = folder_paths.get_folder_paths("emu35")
    subfolders = []
    for path in model_paths:
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    subfolders.append(item)
    return sorted(list(set(subfolders)))

class Emu35Loader:
    @classmethod
    def INPUT_TYPES(s):
        # Get list of subfolders (e.g. "Emu3.5-Image", "vision_tokenizer")
        available_folders = get_emu_subfolders()
        if not available_folders:
            available_folders = ["No folders found in models/emu35"]
            
        return {
            "required": {
                "model_name": (available_folders,),
                "vq_model_name": (available_folders, {"default": "vision_tokenizer" if "vision_tokenizer" in available_folders else available_folders[0]}),
                # "auto" = detect from config (use for pre-quantized NF4 models from HuggingFace)
                # "bf16/fp16/fp32" = load in that precision (for full-precision models)
                # "nf4 (quantize)" = quantize on-the-fly during loading (for full-precision models)
                "precision": (["auto", "bf16", "fp16", "fp32", "nf4 (quantize)"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("EMU_MODEL", "EMU_TOKENIZER", "EMU_VQ")
    RETURN_NAMES = ("model", "tokenizer", "vq_model")
    FUNCTION = "load_model"
    CATEGORY = "Emu3.5"

    def load_model(self, model_name, vq_model_name, precision):
        import json
        
        # Resolve paths
        # We assume the user selected a folder name from the dropdown
        # We need to find which root path contains this folder
        
        def find_folder_path(folder_name):
            for path in folder_paths.get_folder_paths("emu35"):
                full_path = os.path.join(path, folder_name)
                if os.path.exists(full_path):
                    return full_path
            return None

        model_path = find_folder_path(model_name)
        vq_path = find_folder_path(vq_model_name)
        
        if model_path is None:
            raise ValueError(f"Could not find model folder: {model_name}")
        if vq_path is None:
            # Fallback: maybe they typed a path manually or it's a file?
            # For now, strict folder matching based on dropdown
            raise ValueError(f"Could not find VQ folder: {vq_model_name}")

        device = comfy.model_management.get_torch_device()

        # Check if model is pre-quantized by reading config.json
        config_path = os.path.join(model_path, "config.json")
        is_pre_quantized = False
        pre_quant_config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_json = json.load(f)
                if "quantization_config" in config_json:
                    is_pre_quantized = True
                    pre_quant_config = config_json["quantization_config"]
                    print(f"✓ Detected pre-quantized model: {pre_quant_config.get('quant_method', 'unknown')} / {pre_quant_config.get('bnb_4bit_quant_type', 'unknown')}")

        # Determine actual loading strategy
        if precision == "auto":
            if is_pre_quantized:
                print(f"Auto-detected: Loading pre-quantized {pre_quant_config.get('bnb_4bit_quant_type', 'NF4')} model")
                load_mode = "pre_quantized"
                dtype = torch.bfloat16  # Compute dtype for pre-quantized
            else:
                print("Auto-detected: Loading as bf16 (no quantization_config found)")
                load_mode = "bf16"
                dtype = torch.bfloat16
        elif precision == "nf4 (quantize)":
            if is_pre_quantized:
                print("⚠️  Warning: Model appears to already be quantized. Loading as pre-quantized instead of re-quantizing.")
                load_mode = "pre_quantized"
                dtype = torch.bfloat16
            else:
                load_mode = "quantize_nf4"
                dtype = torch.bfloat16
        else:
            # bf16, fp16, fp32
            load_mode = precision
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32

        print(f"Loading Emu3.5 model from {model_path}")
        print(f"  Precision setting: {precision}")
        print(f"  Load mode: {load_mode}")
        print(f"  Compute dtype: {dtype}")

        # Load Config
        try:
            config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
            # Register Emu3 with AutoConfig to prevent tokenizer loading errors
            try:
                from transformers import AutoConfig
                AutoConfig.register("Emu3", Emu3Config)
                print("Registered Emu3Config with AutoConfig.")
            except Exception as e:
                print(f"Warning: Could not register Emu3Config: {e}")
        except Exception:
            # Fallback if Emu3Config is not found (e.g. if repo not in path correctly or using standard config)
            print("Could not load Emu3Config, trying AutoConfig...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # CRITICAL: Use eager attention - sdpa produces noise on Blackwell GPUs (sm_120) + CUDA 12.8
        # This matches the working emu35test/Emu3.5/src/utils/model_utils.py configuration
        attn_impl = "eager"
        print("Using 'eager' attention implementation (sdpa causes corruption on Blackwell GPUs)")

        # Load Model based on mode
        if load_mode == "pre_quantized":
            # Pre-quantized model: DO NOT pass quantization_config, let it auto-detect from saved config
            print("Loading pre-quantized model (no BitsAndBytesConfig - using saved quantization)")
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map="cuda:0",
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
            print("✓ Pre-quantized model loaded successfully")
            
        elif load_mode == "quantize_nf4":
            # On-the-fly quantization of a full-precision model
            print("Quantizing model on-the-fly with NF4...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head", "model.embed_tokens", "model.norm"]  # Match HF model
            )
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=quantization_config,
                device_map="cuda:0",
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
            
            # Verify lm_head is NOT quantized
            if hasattr(model, "lm_head"):
                lm_head_type = str(type(model.lm_head))
                lm_head_dtype = model.lm_head.weight.dtype
                
                print(f"\nNF4 Verification:")
                print(f"  lm_head type: {lm_head_type}")
                print(f"  lm_head dtype: {lm_head_dtype}")
                
                if "Linear4bit" in lm_head_type:
                    print("="*80)
                    print("⚠️  CRITICAL ERROR: lm_head was quantized!")
                    print("⚠️  This will produce GARBAGE outputs (visual noise)")
                    print("⚠️  Your bitsandbytes/transformers version doesn't support skip_modules")
                    print("⚠️  SOLUTION: Use bf16, fp16, or fp32 precision instead")
                    print("="*80)
                    raise ValueError("lm_head quantization detected - cannot proceed with NF4")
                else:
                    print(f"  ✓ lm_head correctly preserved (not quantized)")
        else:
            # Standard precision loading (bf16, fp16, fp32)
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map="cuda:0",  # Force entire model to GPU 0 (no CPU offloading)
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        
        model.eval()
        
        # Enable PyTorch optimizations for Blackwell/Ada GPUs
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')  # Use TF32 for better throughput
        
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # DISABLED: torch.compile + CUDA graphs incompatible with cudaMallocAsync on CUDA 12.8 + Blackwell
        # Error: "cudaMallocAsync does not yet support checkPoolLiveAllocations"
        # TODO: Re-enable when PyTorch/CUDA fixes CUDA graph compatibility
        # try:
        #     import triton
        #     print("Triton detected. Attempting to compile model with torch.compile...")
        #     model = torch.compile(model, mode="max-autotune", fullgraph=False)
        #     print("✓ Model compiled successfully with torch.compile")
        # except ImportError:
        #     print("Triton not found. Skipping torch.compile.")
        # except Exception as e:
        #     print(f"Warning: torch.compile failed: {e}")
        #     print("Model will still work, just not compiled")
        print("torch.compile disabled (incompatible with CUDA 12.8 + Blackwell)")

        # Load Tokenizer
        print(f"Loading tokenizer from {model_path}...")
        
        # Fix for missing tokenization_emu3.py in model folder
        # Transformers requires this file to be present in the model directory for trust_remote_code=True
        try:
            import shutil
            # Use our local patched version instead of the repo version
            repo_tokenizer_file = os.path.join(current_dir, "patched_tokenization_emu3.py")
            dest_tokenizer_file = os.path.join(model_path, "tokenization_emu3.py")
            
            if os.path.exists(repo_tokenizer_file):
                # Always copy to ensure we have the latest patched version (with fallback tokens)
                shutil.copy(repo_tokenizer_file, dest_tokenizer_file)
                print(f"Copied patched tokenizer from {repo_tokenizer_file} to {dest_tokenizer_file}")
            else:
                print(f"Warning: Patched tokenizer not found at {repo_tokenizer_file}")
        except Exception as e:
            print(f"Warning: Could not copy tokenization_emu3.py: {e}")

        # Load tokenizer EXACTLY like the working emu35test/Emu3.5/src/utils/model_utils.py does:
        # Using AutoTokenizer.from_pretrained with special_tokens_file passed explicitly
        try:
            special_tokens_file = os.path.join(model_path, "emu3_vision_tokens.txt")
            print(f"Loading tokenizer with AutoTokenizer.from_pretrained (matching official model_utils.py)...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                special_tokens_file=special_tokens_file,  # Official code passes this explicitly!
                trust_remote_code=True,
                padding_side="left"
            )
            print(f"✓ Tokenizer loaded successfully: {type(tokenizer).__name__}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Trying without special_tokens_file parameter...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
        
        # === CRITICAL: Set tokenizer special token attributes ===
        # The official model_utils.py sets these explicitly - used by multimodal_decode and other functions!
        # These token STRINGS must match what's in emu3_vision_tokens.txt
        tokenizer.bos_token = "<|extra_203|>"
        tokenizer.eos_token = "<|extra_204|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eol_token = "<|extra_200|>"
        tokenizer.eof_token = "<|extra_201|>"
        tokenizer.tms_token = "<|extra_202|>"
        tokenizer.img_token = "<|image token|>"    # Official: <|image token|>
        tokenizer.boi_token = "<|image start|>"    # Official: <|image start|>
        tokenizer.eoi_token = "<|image end|>"      # Official: <|image end|>
        tokenizer.bss_token = "<|extra_100|>"
        tokenizer.ess_token = "<|extra_101|>"
        tokenizer.bog_token = "<|extra_60|>"
        tokenizer.eog_token = "<|extra_61|>"
        tokenizer.boc_token = "<|extra_50|>"
        tokenizer.eoc_token = "<|extra_51|>"
        
        # ============== TOKENIZER VERIFICATION ==============
        print("\n" + "="*80)
        print("TOKENIZER VERIFICATION:")

        special_tokens_to_check = {
            "boi": tokenizer.boi_token,
            "eoi": tokenizer.eoi_token,
            "img": tokenizer.img_token,
            "eol": tokenizer.eol_token,
        }

        for name, token_str in special_tokens_to_check.items():
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            print(f"  {name}: '{token_str}' -> ID {token_id}")

        # CRITICAL: Test visual token mapping
        test_visual_token = "<|visual token 000000|>"
        visual_id_0 = tokenizer.convert_tokens_to_ids(test_visual_token)
        print(f"  visual_0: '{test_visual_token}' -> ID {visual_id_0}")

        if visual_id_0 != 151854:
            print(f"  ⚠️  WARNING: Visual token 0 has WRONG ID!")
            print(f"  ⚠️  Expected 151854, got {visual_id_0}")
            print(f"  ⚠️  This will cause corrupted images!")
        else:
            print(f"  ✓ Visual token mapping is correct")

        # Test decoding
        test_ids = [151854, 151855, 151856]
        decoded_visual = tokenizer.decode(torch.tensor(test_ids), skip_special_tokens=False)
        print(f"  Decode test: {test_ids}")
        print(f"    -> {decoded_visual[:100]}...")

        print("="*80 + "\n")
        # ============== END VERIFICATION ==============
        
        # Load VQ-VAE
        # If vq_path points to a file (like model.ckpt), use its parent directory
        if os.path.isfile(vq_path):
            vq_path = os.path.dirname(vq_path)

        print(f"Loading Vision Tokenizer from {vq_path}...")
        # build_vision_tokenizer expects the directory containing model.ckpt and config.yaml
        vq_model = build_vision_tokenizer("ibq", vq_path, device=device)
        vq_model.eval()
        # Keep VQ model in float32 - converting to bfloat16 causes decode artifacts!
        # vq_model.to(dtype)  # REMOVED - VQ model must stay float32

        return (model, tokenizer, vq_model)

class Emu35VQA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_vqa"
    CATEGORY = "Emu3.5"

    def run_vqa(self, model, tokenizer, vq_model, image, question, max_new_tokens):
        import numpy as np
        from PIL import Image
        
        device = model.device
        
        # Helper to format image string
        def format_image_string(tokenizer, image_tokens):
            h, w = image_tokens.shape
            image_string = ""
            for _h in range(h):
                row_string = ""
                for _w in range(w):
                    row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])
                if _h < h - 1:
                    row_string += tokenizer.eol_token
                image_string += row_string
            
            return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
                image_start=tokenizer.boi_token,
                token_height=h,
                token_width=w,
                image_token=tokenizer.img_token,
                token_str=image_string,
                image_end=tokenizer.eoi_token,
            )

        # 1. Process Image
        # ComfyUI image is [B, H, W, C] float32 0-1
        # Take first image in batch
        img_tensor = image[0] 
        # Convert to [C, H, W] and scale to -1..1 for VQ model
        # VQ model expects [B, C, H, W]
        img_input = (img_tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device, vq_model.dtype)
        
        # 2. Encode Image
        print("Encoding image for VQA...")
        with torch.no_grad():
            _, _, token = vq_model.encode(img_input)
        
        # Reshape tokens
        # token is [B, L] -> need [H, W]
        # Emu VQ downsamples by 16?
        H, W = img_tensor.shape[0], img_tensor.shape[1]
        h_tok, w_tok = H // 16, W // 16
        
        # Ensure token length matches
        if token.shape[1] != h_tok * w_tok:
            # Resize might be needed if image dims aren't multiples of 16
            # For now, assume user provides correct dims or we might crash/need resize logic
            print(f"Warning: Token shape {token.shape} does not match expected {h_tok}x{w_tok}")
            
        token = token[-1].view(h_tok, w_tok)
        image_str = format_image_string(tokenizer, token)

        # 3. Format Prompt
        # Using the 'story' template as a generic assistant template
        template = "<|extra_203|>You are a helpful assistant. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>"
        prompt = template.format(question=question)
        prompt = prompt.replace("<|IMAGE|>", image_str)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        # 4. Generate
        print("Generating answer...")
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False, # Greedy for VQA usually better
        )
        
        with torch.no_grad():
            outputs = model.generate(input_ids, generation_config=gen_config)
            
        # Decode only new tokens
        new_tokens = outputs[:, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        return (response,)

class Emu35Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_image": ("IMAGE",), # For editing/variation
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("image", "text", "reasoning",)
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(self, model, tokenizer, vq_model, prompt, negative_prompt, width, height, steps, cfg_scale, seed, input_image=None):
        device = model.device
        torch.manual_seed(seed)

        # Calculate latent dimensions 
        latent_height = height // 16
        latent_width = width // 16
        
        # =========================================================
        # FIXED PROMPT FORMAT - Must match official Emu3.5 training
        # =========================================================
        # The model expects a chat-style prompt with USER/ASSISTANT roles.
        # The model will generate: BOI -> H*W -> IMG -> visual tokens -> EOI
        # We do NOT put the resolution in the prompt - the logits processor controls it.
        #
        # Official template from configs/example_config_t2i.py:
        # tmpl = "<|extra_203|>You are a helpful assistant for t2i task. USER: {question} ASSISTANT: <|extra_100|>"
        # unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        
        # Build the prompt using official template
        full_prompt = f"<|extra_203|>You are a helpful assistant for t2i task. USER: {prompt} ASSISTANT: <|extra_100|>"
        
        # Unconditional prompt (for CFG negative guidance)
        if negative_prompt and negative_prompt.strip():
            # If user provides negative prompt, use it in the unconditional
            full_negative_prompt = f"<|extra_203|>You are a helpful assistant for t2i task. USER: {negative_prompt} ASSISTANT: <|extra_100|>"
        else:
            # Default: empty user message (standard unconditional)
            full_negative_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        
        print(f"DEBUG: Prompt: {full_prompt[:100]}...")
        print(f"DEBUG: Unconditional: {full_negative_prompt[:100]}...")
        print(f"DEBUG: Target resolution: {latent_height}x{latent_width} latents ({height}x{width} pixels)")
        
        # Prepare Unconditional IDs
        unconditional_ids = tokenizer.encode(full_negative_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Prepare Prompt IDs
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        # Handle Input Image (if provided)
        # If input_image is provided, we need to encode it and prepend/insert tokens.
        # This part depends heavily on Emu 3.5's specific editing implementation.
        # For now, I'll implement basic T2I.
        # If input_image is present, we might need to use the VQ model to encode it.
        
        # Logits Processor
        # This is crucial for Emu 3.5
        logits_processor = LogitsProcessorList()
        
        # Calculate required tokens: H * (W + 1) for EOL tokens + overhead for text, BOI, EOI, etc.
        # The official config uses max_new_tokens=5120 which provides plenty of buffer.
        # The model may generate text response BEFORE the image tokens.
        # For 64x64 latents: 64 * 65 = 4160 image tokens, plus BOI, EOI, IMG, text response overhead
        base_image_tokens = latent_height * (latent_width + 1)  # Image tokens with EOL
        text_overhead = 500  # Buffer for text response before image, BOI, EOI, IMG tokens
        needed_tokens = base_image_tokens + text_overhead
        # Ensure at least 5120 as per official config
        needed_tokens = max(needed_tokens, 5120)
        
        # Resolution tokens
        # Emu3 expects pixel resolution in the resolution string (e.g. "1024*1024")
        # But the model internally works with latents.
        # Let's check if passing pixel resolution fixes the "small image" issue.
        # If the model expects "H*W" where H,W are latents, then "64*64" is correct for 1024px.
        # But the user reports 64x64 output.
        # If I pass "1024*1024", maybe it works?
        # Wait, the logits processor prints "User defined: height: 64, width: 64".
        # This comes from self.target_height/width in DummyConfig.
        # In DummyConfig, I set self.target_height = latent_height (64).
        # So the logits processor thinks the target is 64x64.
        # Does the logits processor expect pixels or latents?
        # If it expects pixels, I should pass height/width.
        
        # Let's try passing PIXEL dimensions to DummyConfig and res_string.
        
        class DummyConfig:
            def __init__(self):
                self.classifier_free_guidance = cfg_scale
                self.image_area = width * height
                self.h = latent_height
                self.w = latent_width
                self.target_height = latent_height
                self.target_width = latent_width
                self.image_cfg_scale = 1.0 # Default from config.py
                self.unconditional_type = "no_text" # Default from config.py
                
                # Special tokens for generation_utils
                self.special_token_ids = {
                    "PAD": tokenizer.pad_token_id,
                    "EOS": tokenizer.eos_token_id,
                    "BOS": tokenizer.bos_token_id,
                }

                # Sampling params - MUST match official configs/example_config_t2i.py
                self.sampling_params = {
                    "use_cache": True,
                    # Text token sampling (more constrained)
                    "text_top_k": 1024,
                    "text_top_p": 0.9,
                    "text_temperature": 1.0,
                    # Image token sampling (more diverse) 
                    "image_top_k": 5120,  # Official uses 5120, not 10240!
                    "image_top_p": 1.0,
                    "image_temperature": 1.0,
                    # General defaults
                    "top_k": 131072,
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "num_beams_per_group": 1,
                    "num_beam_groups": 1,
                    "diversity_penalty": 0.0,
                    "max_new_tokens": needed_tokens,
                    "guidance_scale": 1.0,  # This is different from CFG scale
                    # Enable differential sampling - CRITICAL for quality!
                    "use_differential_sampling": True,
                    "do_sample": True,
                    "num_beams": 1,
                }
                
        dummy_cfg = DummyConfig()

        try:
            processor = build_logits_processor(
                cfg=dummy_cfg,
                unconditional_ids=unconditional_ids,
                model=model,
                tokenizer=tokenizer,
                force_same_image_size=True
            )
            logits_processor.append(processor)
            print(f"DEBUG: LogitsProcessor successfully built with CFG scale {cfg_scale}")
        except Exception as e:
            print(f"Warning: Error building logits processor: {e}")
            print("Continuing without custom logits processor - CFG will NOT be applied!")
            # Continue without it - may be faster without overhead

        # Generation Config - match official generation_utils.py
        # The official code does: GenerationConfig(**cfg.sampling_params, pad_token_id=..., eos_token_id=...)
        generation_config = GenerationConfig(
            **dummy_cfg.sampling_params,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Fix for 'NoneType' object has no attribute 'transformers_version'
        # The model instance might not have generation_config set, which GenerationMixin expects.
        if not hasattr(model, "generation_config") or model.generation_config is None:
            model.generation_config = generation_config
        
        # Check for generate method
        # Emu3ForCausalLM inherits from Emu3PreTrainedModel -> PreTrainedModel
        # PreTrainedModel has a 'generate' method, but sometimes it's not exposed if the mixin isn't correct
        # or if the object is wrapped.
        
        # Explicitly check if we can use the GenerationMixin's generate
        from transformers.generation.utils import GenerationMixin
        
        # If generate is missing, we might need to call the superclass method or bind it?
        # But PreTrainedModel *is* a GenerationMixin.
        
        if not hasattr(model, "generate"):
            print(f"WARNING: Model object {type(model)} has no 'generate' method.")
            if hasattr(model, "module") and hasattr(model.module, "generate"):
                print("Found 'generate' in model.module. Using that.")
                model = model.module
            else:
                print("Could not find 'generate' method. Attempting to use transformers.generation.utils.GenerationMixin.generate directly.")
                # This is a hack, but if the class hierarchy is messed up, we can try to bind the method
                import types
                
                # Bind all missing GenerationMixin methods
                # _extract_generation_mode_kwargs is a private method needed by generate
                # We can try to mixin the class dynamically
                
                # Option 1: Add GenerationMixin to the bases
                cls = model.__class__
                if GenerationMixin not in cls.__bases__:
                    print("Adding GenerationMixin to model class bases...")
                    cls.__bases__ = (GenerationMixin,) + cls.__bases__
                
                # Option 2: If that fails, manually bind methods (less reliable)
                if not hasattr(model, "generate"):
                     model.generate = types.MethodType(GenerationMixin.generate, model)

        # Enable optimizations for faster inference
        # Use custom batched generation for speed
        # print("Using Batched Generation for speed...")
        
        # REVERTING TO STANDARD GENERATION
        # The custom loop is causing issues with image size/coherence because it bypasses the LogitsProcessor state machine.
        # We will use model.generate() which correctly uses the LogitsProcessor we built above.
        
        print("Using standard model.generate() for correctness... (Latent Size: {}x{})".format(latent_height, latent_width))
        
        # Streamer for progress bar
        streamer = ThrottledStreamer(needed_tokens)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer
            )
            
        # DEBUG: Print FULL output including input
        full_output = outputs[0].tolist()
        print(f"DEBUG: Full output length: {len(full_output)}, Input length: {input_ids.shape[1]}")
        print(f"DEBUG: Full output first 30 tokens: {full_output[:30]}")
        print(f"DEBUG: Full output tokens around input boundary: {full_output[input_ids.shape[1]-5:input_ids.shape[1]+10]}")
        
        # Check for special tokens in FULL output
        print(f"DEBUG: BOI (151852) in full output: {BOI_TOKEN_ID in full_output}")
        print(f"DEBUG: EOI (151853) in full output: {EOI_TOKEN_ID in full_output}")
        print(f"DEBUG: IMG (151851) in full output: {IMG_TOKEN_ID in full_output}")
            
        # Decode only new tokens
        # outputs contains [input_ids + generated_tokens]
        # We need to extract just the generated part for decoding
        generated_tokens_tensor = outputs[:, input_ids.shape[1]:]
        
        # DEBUG: Print raw generated token IDs (first 20)
        raw_ids = generated_tokens_tensor[0].tolist()
        print(f"DEBUG: Raw generated token IDs (first 20): {raw_ids[:20]}")
        print(f"DEBUG: Total tokens generated: {len(raw_ids)}")
        print(f"DEBUG: Expected tokens for {latent_height}x{latent_width} image: ~{latent_height * (latent_width + 1)}")
        
        # Convert to list for compatibility with existing decode logic below
        generated_tokens = generated_tokens_tensor[0].tolist()
        
        """
        # Prepare batched input
        # Pad inputs to same length
        p_len = input_ids.shape[1]
        u_len = unconditional_ids.shape[1]
        max_len = max(p_len, u_len)
        
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Left pad
        if p_len < max_len:
            pad = torch.full((1, max_len - p_len), pad_id, device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([pad, input_ids], dim=1)
        if u_len < max_len:
            pad = torch.full((1, max_len - u_len), pad_id, device=device, dtype=unconditional_ids.dtype)
            unconditional_ids = torch.cat([pad, unconditional_ids], dim=1)
            
        # Batch them: [Uncond, Cond]
        batched_input = torch.cat([unconditional_ids, input_ids], dim=0)
        
        # Constants
        BOI_TOKEN = 151852
        IMG_TOKEN = 151851
        EOI_TOKEN = 151853
        EOL_TOKEN = 151846 # <|extra_200|>
        
        # Resolution tokens
        # Use PIXEL resolution string
        res_string = f"{latent_height}*{latent_width}"
        res_tokens = tokenizer.encode(res_string, add_special_tokens=False)
        res_tokens_tensor = torch.tensor(res_tokens, device=device).unsqueeze(0).repeat(2, 1)
        
        # Generation Loop
        with torch.inference_mode():
            # 1. Prefill
            past_key_values = None
            outputs = model(batched_input, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # We expect the model to output BOI next (or very soon)
            # But we can force it or let it generate.
            # For safety, let's just start the loop.
            
            cur_len = 0
            max_gen = generation_config.max_new_tokens
            
            # State tracking
            in_image = False
            vis_idx = 0
            
            # Streamer
            streamer = ThrottledStreamer(max_gen)
            
            generated_tokens = []
            
            for _ in range(max_gen):
                # Split logits
                neg_logits = next_token_logits[0]
                pos_logits = next_token_logits[1]
                
                # CFG
                # Only apply CFG if we are in image mode (generating visual tokens)
                # Or should we apply it always? Emu3 applies it only for visual tokens usually.
                # But let's apply it always for consistency, or check state.
                
                # Check if we just generated BOI
                # We don't know what we generated yet, we are choosing now.
                
                # Logic:
                # 1. Apply CFG
                # 2. Force tokens if needed (Resolution, EOL, EOI)
                # 3. Sample
                
                scores = neg_logits + cfg_scale * (pos_logits - neg_logits)
                
                # Force Resolution Logic
                # If we haven't started image yet, wait for BOI
                if not in_image:
                    # Let it generate naturally until BOI
                    # But we can force BOI if it takes too long?
                    # No, let's trust the model.
                    pass
                else:
                    # We are in image block
                    # Check if we need to force Resolution string
                    # The model usually generates: BOI -> H -> * -> W -> IMG -> pixels
                    pass

                # Apply Sampling (Temperature)
                if generation_config.temperature != 1.0 and generation_config.temperature > 0:
                    scores = scores / generation_config.temperature
                
                # Top K
                if generation_config.top_k > 0:
                    v, _ = torch.topk(scores, min(generation_config.top_k, scores.size(-1)))
                    scores[scores < v[..., -1, None]] = -float('inf')
                    
                probs = F.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (1,)
                
                # Force Logic Override
                # If we just generated BOI, force resolution sequence
                if len(generated_tokens) > 0 and generated_tokens[-1] == BOI_TOKEN:
                    in_image = True
                    # Force resolution tokens next
                    # We inject them one by one? Or all at once?
                    # We must inject one by one to update cache.
                    # Actually, we can just append them to history and run forward?
                    # But we are in a loop.
                    # Let's handle this by overriding `next_token`
                    pass
                
                # Simplified Logic:
                # We will just let the model run, but if we are in image mode, we force EOL/EOI
                
                # Check for BOI in history
                if not in_image and len(generated_tokens) > 0 and generated_tokens[-1] == BOI_TOKEN:
                    in_image = True
                    # We need to inject the resolution tokens NOW
                    # Instead of sampling, we force the next tokens to be the resolution string
                    # But we can't inject multiple tokens in one step of this loop easily without complex logic.
                    # So we will just force them one by one in subsequent steps?
                    # No, that's hard to track.
                    
                    # Better approach:
                    # When BOI is detected, we interrupt, feed the whole resolution string + IMG token,
                    # update cache, and resume.
                    
                    # Feed Res String + IMG
                    to_feed = torch.cat([res_tokens_tensor, torch.tensor([[IMG_TOKEN], [IMG_TOKEN]], device=device)], dim=1)
                    
                    # Forward pass for forced tokens
                    outputs = model(to_feed, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Add to generated
                    for t in res_tokens:
                        generated_tokens.append(t)
                    generated_tokens.append(IMG_TOKEN)
                    
                    # Continue loop
                    continue

                if in_image:
                    # We are generating pixels
                    # Check for EOL / EOI
                    # vis_idx counts pixels (tokens after IMG)
                    # We need to know how many pixels we have generated
                    # Find index of IMG
                    try:
                        img_idx = generated_tokens.index(IMG_TOKEN)
                        current_vis_count = len(generated_tokens) - img_idx - 1
                        
                        # Next token index is current_vis_count + 1
                        next_idx = current_vis_count + 1
                        
                        # EOI
                        if next_idx == latent_height * (latent_width + 1):
                            next_token = torch.tensor([EOI_TOKEN], device=device)
                        # EOL
                        elif next_idx % (latent_width + 1) == 0:
                            next_token = torch.tensor([EOL_TOKEN], device=device)
                            
                    except ValueError:
                        pass

                # Append to list
                token_val = next_token.item()
                generated_tokens.append(token_val)
                
                # Prepare input for next step
                # Feed same token to both batch items
                next_input = torch.cat([next_token, next_token], dim=0).unsqueeze(1)
                
                outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # Streamer
                streamer.put(next_input)
                
                # Stop if EOI
                if token_val == EOI_TOKEN:
                    break

        # Decode
        # generated_tokens is a list of ints
        # Convert to tensor (1, L)
        generated_tokens_tensor = torch.tensor(generated_tokens, device=device).unsqueeze(0)
        """
        
        print("Decoding response...")
        
        # Ensure tokenizer has eol_token attribute if missing
        if not hasattr(tokenizer, "eol_token"):
            # Emu3 uses <|extra_200|> (ID 151846) as end of line token
            # We must match what the model generates
            tokenizer.eol_token = "<|extra_200|>"
            print(f"DEBUG: Set tokenizer.eol_token to {tokenizer.eol_token}")
        
        # Extract text response before image (everything before <|image start|>)
        text_response = ""
            
        # decode_image expects a string, but we have tokens.
        # We need to decode tokens to string first?
        # Looking at generation_utils.py: decode_image(image_string, ...)
        # It splits by eol_token.
        
        # Let's try to decode the tokens to text first, as decode_image expects a string
        # We use errors='ignore' to prevent crashing on unknown tokens if the fallback above isn't enough
        try:
            decoded_text = tokenizer.decode(generated_tokens_tensor[0], skip_special_tokens=False)
        except Exception as e:
            print(f"Warning: Tokenizer decode failed ({e}). Trying with errors='ignore'...")
            decoded_text = None

            # Attempt to re-run decode with explicit errors='ignore' if the tokenizer supports it
            try:
                decoded_text = tokenizer.decode(
                    generated_tokens_tensor[0],
                    skip_special_tokens=False,
                    errors="ignore",
                )
            except Exception as inner:
                print(f"Secondary decode attempt failed ({inner}). Trying raw tiktoken decode...")

            if decoded_text is None and hasattr(tokenizer, "tokenizer"):
                try:
                    decoded_text = tokenizer.tokenizer.decode(
                        generated_tokens_tensor[0].tolist(),
                        errors="ignore",
                    )
                except Exception as inner:
                    print(f"Raw tiktoken decode failed ({inner}).")

            if decoded_text is None:
                # As a last resort, strip out any token IDs that are outside the known vocab to avoid hard crashes
                fallback_ids = [tok for tok in generated_tokens_tensor[0].tolist() if tok in getattr(tokenizer, "decoder", {})]
                if fallback_ids:
                    decoded_text = tokenizer.decode(fallback_ids, skip_special_tokens=False, errors="ignore")

            if decoded_text is None:
                raise e

        # NEW CODE - use official decoding path
        try:
            # Decode FULL output to string (including input)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            print(f"Decoded output length: {len(output_text)} chars")
            
            # Use official multimodal_decode (handles multiple images/text)
            mm_output = multimodal_decode(output_text, tokenizer, vq_model)
            
            # Extract image, text, and reasoning (CoT)
            image = None
            text_response = ""
            reasoning = ""
            for item_type, item_data in mm_output:
                if item_type == "image" and item_data is not None:
                    image = item_data
                elif item_type == "text":
                    text_response += item_data
                elif item_type == "global_cot":
                    reasoning += f"[Global] {item_data}\n"
                elif item_type == "image_cot":
                    reasoning += f"[Image] {item_data}\n"
                 
            if image is None:
                 raise ValueError("Failed to decode image from tokens.")
                 
            # Convert PIL to Tensor (B, H, W, C) -> (B, C, H, W) for Comfy?
            # Comfy expects (B, H, W, C) in range [0, 1]
            
            import numpy as np
            from PIL import Image
            
            if isinstance(image, Image.Image):
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # Add batch dim
                return (image_tensor, text_response, reasoning)
            else:
                 # If it's not a PIL image, maybe it's already a tensor or something else?
                 # But decode_image in repo returns PIL Image.
                 print(f"Warning: decode_image returned {type(image)}")
                 empty_img = torch.zeros((1, height, width, 3))
                 return (empty_img, "", "")
            
        except Exception as e:
            print(f"Error decoding image: {e}")
            import traceback
            traceback.print_exc()
            empty_img = torch.zeros(1, height, width, 3)
            return (empty_img, "", "")

class Emu35ClearCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": ("*", {}), # Dummy input to trigger execution
            },
            "optional": {
                "model": ("EMU_MODEL",),
                "vq_model": ("EMU_VQ",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "clear_cache"
    CATEGORY = "Emu3.5"
    OUTPUT_NODE = True

    def clear_cache(self, any_input, model=None, vq_model=None):
        import gc
        
        print("Clearing Emu3.5 models from VRAM...")
        
        if model is not None:
            try:
                model.to("cpu")
                del model
                print("Unloaded LLM model.")
            except:
                pass
                
        if vq_model is not None:
            try:
                vq_model.to("cpu")
                del vq_model
                print("Unloaded VQ model.")
            except:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        print("VRAM cleared.")
        return ()

# Define ThrottledStreamer locally if not exists
class ThrottledStreamer(ComfyStreamer):
    def put(self, value):
        self.counter += 1
        if self.counter % 64 == 0 or self.counter == self.total_steps:
            self.pbar.update(64)
        return value


class Emu35OfficialT2I:
    """
    Text-to-Image node using EXACT official Emu3.5 code structure.
    Matches example_config_t2i.py exactly.
    """
    
    # Official aspect ratios from example_config_t2i.py
    ASPECT_RATIOS = {
        "4:3": (55, 73),    # 880x1168
        "21:9": (41, 97),   # 656x1552
        "16:9": (47, 85),   # 752x1360
        "3:2": (52, 78),    # 832x1248
        "1:1": (64, 64),    # 1024x1024
        "3:4": (73, 55),    # 1168x880
        "9:16": (85, 47),   # 1360x752
        "2:3": (78, 52),    # 1248x832
        "default": (55, 73), # 880x1168 (same as 4:3)
    }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat sitting on a windowsill"}),
                "aspect_ratio": (["default", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"],),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("image", "text", "reasoning",)
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(self, model, tokenizer, vq_model, prompt, aspect_ratio, cfg_scale, seed):
        import gc
        
        device = "cuda"
        torch.manual_seed(seed)
        
        # Special token IDs (from official config)
        BOS = 151849  # <|extra_203|>
        EOS = 151850  # <|extra_204|>
        PAD = 151643  # <|endoftext|>
        BOI = 151852
        EOI = 151853
        IMG = 151851
        
        # Get target size from aspect ratio (EXACTLY like official config)
        target_height, target_width = self.ASPECT_RATIOS.get(aspect_ratio, (55, 73))
        
        # Match working my_test_config.py: max_new_tokens=5120
        max_new_tokens = 5120
        
        print(f"[Emu35OfficialT2I] Aspect: {aspect_ratio} -> {target_height}x{target_width} latents ({target_height*16}x{target_width*16} px)")
        
        # === Build config object matching my_test_config.py EXACTLY ===
        class OfficialConfig:
            def __init__(self, cfg_scale, target_h, target_w):
                # From my_test_config.py
                self.classifier_free_guidance = cfg_scale
                self.unconditional_type = "no_text"
                self.image_cfg_scale = 1.0
                
                # Target size from aspect ratio
                self.target_height = target_h
                self.target_width = target_w
                
                # Sampling params - matching working my_test_config.py exactly
                self.sampling_params = dict(
                    use_cache=True,
                    # text token sampling config
                    text_top_k=1024,
                    text_top_p=0.9,
                    text_temperature=1.0,
                    # image token sampling config - matching my_test_config.py
                    image_top_k=5120,
                    image_top_p=1.0,
                    image_temperature=1.0,
                    # general config
                    top_k=131072,
                    top_p=1.0,
                    temperature=1.0,
                    num_beams_per_group=1,
                    num_beam_groups=1,
                    diversity_penalty=0.0,
                    max_new_tokens=5120,  # matching my_test_config.py
                    guidance_scale=1.0,
                    # enable differential sampling
                    use_differential_sampling=True,
                    do_sample=True,
                    num_beams=1,
                )
        
        cfg = OfficialConfig(cfg_scale, target_height, target_width)
        
        # === Build prompts matching official template ===
        # Official t2i template (no image input)
        template = "<|extra_203|>You are a helpful assistant for t2i task. USER: {question} ASSISTANT: <|extra_100|>"
        unc_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        
        full_prompt = template.format(question=prompt)
        
        print(f"[Emu35OfficialT2I] Prompt: {full_prompt[:100]}...")
        print(f"[Emu35OfficialT2I] CFG: {cfg_scale}, max_new_tokens: {max_new_tokens}")
        
        # === Encode prompts ===
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Add BOS if not present (official does this)
        if input_ids[0, 0] != BOS:
            bos_tensor = torch.tensor([[BOS]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        
        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        print(f"[Emu35OfficialT2I] Input shape: {input_ids.shape}, Unconditional shape: {unconditional_ids.shape}")
        
        # === Build logits processor (official function) ===
        print("[Emu35OfficialT2I] Building logits processor...")
        print(f"[Emu35OfficialT2I] cfg.target_height={cfg.target_height}, cfg.target_width={cfg.target_width}")
        print(f"[Emu35OfficialT2I] cfg.sampling_params max_new_tokens={cfg.sampling_params['max_new_tokens']}")
        
        lp = build_logits_processor(
            cfg=cfg,
            unconditional_ids=unconditional_ids,
            model=model,
            tokenizer=tokenizer,
            full_unconditional_ids=None,
            force_same_image_size=True,
        )
        
        # Wrap to count calls
        original_call = lp.__call__
        call_count = [0]
        def counting_call(input_ids, scores):
            call_count[0] += 1
            if call_count[0] <= 5:
                print(f"[DEBUG] LogitsProcessor called #{call_count[0]}, last token: {input_ids[0, -1].item()}")
            return original_call(input_ids, scores)
        lp.__call__ = counting_call
        
        logits_processor = LogitsProcessorList()
        logits_processor.append(lp)
        
        # === Build stopping criteria to stop after first image ===
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopAfterFirstImage(StoppingCriteria):
            """Stop generation after we see EOI (end of image) token."""
            def __init__(self, eoi_token_id):
                self.eoi_token_id = eoi_token_id
                self.found_eoi = False
                
            def __call__(self, input_ids, scores, **kwargs):
                # Early return if we already found EOI (avoids repeated checks)
                if self.found_eoi:
                    return True
                # GPU-native check - no expensive copy to CPU
                # This is called ~4000+ times during generation, so avoiding .tolist() is critical
                if (input_ids[0] == self.eoi_token_id).any().item():
                    self.found_eoi = True
                    print("[Emu35OfficialT2I] Found EOI token, stopping generation")
                    return True
                return False
        
        stopping_criteria = StoppingCriteriaList([StopAfterFirstImage(EOI)])
        
        # === Build generation config (official way) ===
        generation_config = GenerationConfig(
            **cfg.sampling_params,
            pad_token_id=PAD,
            eos_token_id=EOS,
        )
        
        # Fix for transformers 4.50+: ensure model has a generation_config
        if model.generation_config is None:
            model.generation_config = GenerationConfig(
                bos_token_id=BOS,
                eos_token_id=EOS,
                pad_token_id=PAD,
            )
            print("[Emu35OfficialT2I] Created default generation_config for model")
        
        # === Generate with progress tracking ===
        print("[Emu35OfficialT2I] Starting generation...")
        print(f"[Emu35OfficialT2I] generation_config.max_new_tokens={generation_config.max_new_tokens}")
        
        # Create a streamer for progress updates
        # For a 55x73 image: 55 rows * (73 visual tokens + 1 EOL) + BOI + IMG + dimensions + EOI ≈ 4100 tokens
        expected_tokens = target_height * (target_width + 1) + 50  # visual tokens + overhead
        streamer = ThrottledStreamer(expected_tokens)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )
        
        # After generation completes
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation complete. Output shape: {outputs.shape}")
        print(f"[PERFORMANCE] Generation took {generation_time:.2f} seconds")
        print(f"[PERFORMANCE] Tokens/second: {outputs.shape[1] / generation_time:.2f}")

        # ============== TOKEN ANALYSIS ==============
        print("\n" + "="*80)
        print("TOKEN ANALYSIS:")
        all_tokens = outputs[0].tolist()
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Generated tokens: {len(all_tokens) - input_ids.shape[1]}")

        # Check for special tokens
        # BOI, EOI, IMG, EOL already defined above
        has_boi = BOI in all_tokens
        has_eoi = EOI in all_tokens
        print(f"Has BOI: {has_boi}, Has EOI: {has_eoi}")

        if has_boi and has_eoi:
            boi_idx = all_tokens.index(BOI)
            eoi_idx = all_tokens.index(EOI)
            
            # Show tokens after BOI (should be resolution + IMG)
            context = all_tokens[boi_idx:min(boi_idx+20, len(all_tokens))]
            context_str = tokenizer.decode(torch.tensor(context), skip_special_tokens=False)
            print(f"After BOI: {context_str}")
            
            # Count visual tokens (>= 151854)
            visual_tokens = [t for t in all_tokens[boi_idx+1:eoi_idx] if t >= 151854]
            print(f"Visual tokens found: {len(visual_tokens)}")
            print(f"Expected: {target_height * target_width}")
            
            if visual_tokens:
                # Show first 10 visual token INDICES (subtract 151854)
                first_10_indices = [t - 151854 for t in visual_tokens[:10]]
                max_index = max([t - 151854 for t in visual_tokens])
                print(f"First 10 indices: {first_10_indices}")
                print(f"Max index: {max_index}")
                
                # Check if any indices are out of VQ codebook range
                if hasattr(vq_model, 'quantize') and hasattr(vq_model.quantize, 'n_embed'):
                    n_embed = vq_model.quantize.n_embed
                    print(f"VQ codebook size: {n_embed}")
                    if max_index >= n_embed:
                        print(f"  ⚠️  WARNING: Max index ({max_index}) >= codebook size ({n_embed})!")
        else:
            print("  ⚠️  ERROR: Missing BOI or EOI tokens!")

        print("="*80 + "\n")
        # ============== END TOKEN ANALYSIS ==============
        
        # === Free model memory before decoding ===
        print("[Emu35OfficialT2I] Freeing model memory for VQ decode...")
        del logits_processor
        
        # Move model to CPU to free VRAM for VQ decode
        # Note: This modifies the model in-place, user will need to reload
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"[Emu35OfficialT2I] GPU memory after cleanup: {mem_after:.2f} GB")
        
        # === Decode output using OFFICIAL multimodal_decode ===
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"[Emu35OfficialT2I] Decoded output length: {len(output_text)}")
        
        # Use the official multimodal_decode function from the repo
        try:
            mm_output = multimodal_decode(output_text, tokenizer, vq_model)
            print(f"[Emu35OfficialT2I] multimodal_decode returned {len(mm_output)} items")
            
            # Extract image, text, and reasoning (CoT) from multimodal output
            image = None
            text_response = ""
            reasoning = ""
            for item_type, item_data in mm_output:
                print(f"[Emu35OfficialT2I] Found item type: {item_type}")
                if item_type == "image" and item_data is not None:
                    # item_data is a PIL Image from official decode_image
                    print(f"[Emu35OfficialT2I] PIL Image size: {item_data.size}, mode: {item_data.mode}")
                    img_np = np.array(item_data).astype(np.float32) / 255.0
                    print(f"[Emu35OfficialT2I] numpy array shape: {img_np.shape}, dtype: {img_np.dtype}")
                    image = torch.from_numpy(img_np).unsqueeze(0)
                    print(f"[Emu35OfficialT2I] Output tensor shape: {image.shape} (should be [1, H, W, 3])")
                elif item_type == "text":
                    text_response += item_data
                elif item_type == "global_cot":
                    reasoning += f"[Global] {item_data}\n"
                elif item_type == "image_cot":
                    reasoning += f"[Image] {item_data}\n"
            
            if image is not None:
                return (image, text_response, reasoning)
                    
            print("[Emu35OfficialT2I] ERROR: No image found in multimodal_decode output")
        except Exception as e:
            print(f"[Emu35OfficialT2I] ERROR in multimodal_decode: {e}")
            import traceback
            traceback.print_exc()
        
        # Return empty image on failure (match target size)
        pixel_height, pixel_width = target_height * 16, target_width * 16
        empty_img = torch.zeros(1, pixel_height, pixel_width, 3)
        return (empty_img, "", "")


NODE_CLASS_MAPPINGS = {
    "Emu35Loader": Emu35Loader,
    "Emu35Sampler": Emu35Sampler,
    "Emu35VQA": Emu35VQA,
    "Emu35ClearCache": Emu35ClearCache,
    "Emu35OfficialT2I": Emu35OfficialT2I,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35Loader": "Emu 3.5 Loader",
    "Emu35Sampler": "Emu 3.5 Sampler",
    "Emu35VQA": "Emu 3.5 VQA",
    "Emu35ClearCache": "Emu 3.5 Clear Cache",
    "Emu35OfficialT2I": "Emu 3.5 Official T2I (Auto Size)",
}
