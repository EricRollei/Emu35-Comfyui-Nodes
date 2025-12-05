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
        from src.utils.generation_utils import build_logits_processor
        # We use our local patched tokenizer instead of the repo one
        try:
            # Ensure current dir is in sys.path
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            from patched_tokenization_emu3 import Emu3Tokenizer
        except ImportError as e:
            Emu3Tokenizer = None
            print(f"Could not import patched Emu3Tokenizer: {e}. Will rely on AutoTokenizer.")
            
    except ImportError as e:
        print(f"Error importing Emu3.5 modules: {e}")
        print("Please ensure the Emu3.5 repository is cloned into 'Emu3_5_repo' inside this node's directory.")

# Local implementation of decode_image to avoid dependency on repo updates
import re
from typing import List
from PIL import Image
import numpy as np

# Constants for token IDs
BOI_TOKEN_ID = 151852  # <|begin_of_image|>
EOI_TOKEN_ID = 151853  # <|end_of_image|>
IMG_TOKEN_ID = 151851  # <|image|>
EOL_TOKEN_ID = 151846  # <|extra_200|> (end of line)
VISUAL_TOKEN_START = 151855  # First visual token ID

def decode_image_from_tokens(token_ids, vision_tokenizer):
    """Decode image directly from token IDs (not string)."""
    # Find BOI and EOI positions
    try:
        boi_idx = token_ids.index(BOI_TOKEN_ID)
        eoi_idx = token_ids.index(EOI_TOKEN_ID)
    except ValueError:
        print("DEBUG: Could not find BOI or EOI tokens in sequence")
        return None
    
    # Extract tokens between BOI and EOI
    image_tokens = token_ids[boi_idx + 1:eoi_idx]
    
    # Split by EOL to get rows
    rows = []
    current_row = []
    
    for tok in image_tokens:
        if tok == EOL_TOKEN_ID:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif tok >= VISUAL_TOKEN_START:
            # Convert token ID to visual token index
            visual_idx = tok - VISUAL_TOKEN_START
            current_row.append(visual_idx)
        # Ignore IMG token and resolution numbers
    
    if current_row:  # Add last row if no trailing EOL
        rows.append(current_row)
    
    if not rows:
        print("DEBUG: No visual tokens found")
        return None
    
    print(f"DEBUG: First 10 visual token indices: {rows[0][:10] if rows[0] else []}")
    
    # Determine target width
    from collections import Counter
    widths = [len(r) for r in rows]
    target_width = Counter(widths).most_common(1)[0][0]
    
    # Pad/truncate rows to same width
    final_image = []
    for row in rows:
        if len(row) == target_width:
            final_image.append(row)
        elif len(row) > target_width:
            final_image.append(row[:target_width])
        else:
            pad_token = row[-1] if row else 0
            final_image.append(row + [pad_token] * (target_width - len(row)))
    
    # Convert to tensor
    image = torch.tensor(
        final_image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
    )
    
    min_tok = image.min().item()
    max_tok = image.max().item()
    print(f"DEBUG: Visual Token Range: [{min_tok}, {max_tok}]")
    
    h, w = image.shape
    
    # Get embedding dim
    if hasattr(vision_tokenizer, "quantize") and hasattr(vision_tokenizer.quantize, "e_dim"):
        embed_dim = vision_tokenizer.quantize.e_dim
        n_embed = getattr(vision_tokenizer.quantize, "n_embed", 262144)
        print(f"DEBUG: VQ-VAE Config - embed_dim: {embed_dim}, n_embed: {n_embed}")
        
        if max_tok >= n_embed:
            print(f"WARNING: Max token ID ({max_tok}) exceeds VQ-VAE codebook size ({n_embed})!")
    else:
        embed_dim = 256
        print(f"Warning: Could not determine embed_dim, using default {embed_dim}")
    
    # Decode to image
    image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, embed_dim)).float()
    image = image[0].permute(1, 2, 0)
    image = Image.fromarray(
        ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    )
    return image

def decode_image(image_string, tokenizer, vision_tokenizer):
    image: List[List[int]] = []
    # Fix for 'Emu3Tokenizer' object has no attribute 'eol_token'
    # The tokenizer might not have eol_token set as an attribute, but it should be in special_tokens
    eol_token = getattr(tokenizer, "eol_token", "<|eol|>")
    
    # Extract only the image region between BOI and EOI
    boi_token = "<|begin_of_image|>"
    eoi_token = "<|end_of_image|>"
    
    # Find the image content
    boi_idx = image_string.find(boi_token)
    eoi_idx = image_string.find(eoi_token)
    
    if boi_idx == -1 or eoi_idx == -1 or boi_idx >= eoi_idx:
        print(f"DEBUG: Could not find image markers in decoded string. BOI: {boi_idx}, EOI: {eoi_idx}")
        return None
    
    # Extract substring between BOI and EOI (excluding the markers themselves)
    image_content = image_string[boi_idx + len(boi_token):eoi_idx]
    
    # Split by EOL tokens
    image_rows = re.split(re.escape(eol_token), image_content)
    
    # First pass: collect all valid rows
    raw_rows = []
    debug_first_tokens = []
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            raw_rows.append(row_token)
            if len(debug_first_tokens) < 10:
                debug_first_tokens.extend(row_token[:10])
    
    if debug_first_tokens:
        print(f"DEBUG: First 10 decoded visual token indices: {debug_first_tokens[:10]}")
            
    if not raw_rows:
        return None

    # Determine target width (use the most common width or the max width)
    # Using max width is safer to avoid cutting off data, but padding is needed
    widths = [len(r) for r in raw_rows]
    if not widths:
        return None
        
    # Use the most frequent width as the target, assuming outliers are errors
    from collections import Counter
    target_width = Counter(widths).most_common(1)[0][0]
    
    # Second pass: pad or truncate
    final_image = []
    for row in raw_rows:
        if len(row) == target_width:
            final_image.append(row)
        elif len(row) > target_width:
            final_image.append(row[:target_width])
        else:
            # Pad with the last token of the row or 0
            # Using the last token is visually safer than black (0)
            pad_token = row[-1] if row else 0
            final_image.append(row + [pad_token] * (target_width - len(row)))
            
    try:
        image = torch.tensor(
            final_image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
        )
        
        # DEBUG: Check token range
        min_tok = image.min().item()
        max_tok = image.max().item()
        print(f"DEBUG: Visual Token Range: [{min_tok}, {max_tok}]")
        
        h, w = image.shape
        
        # Get embedding dim dynamically from the quantizer
        # The quantizer is usually at vision_tokenizer.quantize
        if hasattr(vision_tokenizer, "quantize") and hasattr(vision_tokenizer.quantize, "e_dim"):
            embed_dim = vision_tokenizer.quantize.e_dim
            n_embed = getattr(vision_tokenizer.quantize, "n_embed", "Unknown")
            print(f"DEBUG: VQ-VAE Config - embed_dim: {embed_dim}, n_embed: {n_embed}")
            
            if isinstance(n_embed, int) and max_tok >= n_embed:
                print(f"WARNING: Max token ID ({max_tok}) exceeds VQ-VAE codebook size ({n_embed})!")
                # Optional: Clamp or modulo?
                # image = image % n_embed
        else:
            embed_dim = 256 # Fallback if not found
            print(f"Warning: Could not determine embed_dim from vision_tokenizer, using default {embed_dim}")

        image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, embed_dim)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(
            ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        return image
    except Exception as ex:
        print(f"decode image failed {ex}")
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
                "precision": (["bf16", "fp16", "fp32", "nf4"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("EMU_MODEL", "EMU_TOKENIZER", "EMU_VQ")
    RETURN_NAMES = ("model", "tokenizer", "vq_model")
    FUNCTION = "load_model"
    CATEGORY = "Emu3.5"

    def load_model(self, model_name, vq_model_name, precision):
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
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32

        print(f"Loading Emu3.5 model from {model_path} with precision {precision}...")

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

        # Check for Flash Attention
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            print("Flash Attention not found. Using 'sdpa' (PyTorch Scaled Dot Product Attention).")
            attn_impl = "sdpa"

        # Load Model
        if precision == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head", "embed_tokens"] # Skip head and embeddings for stability
            )
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=quantization_config,
                device_map="cuda:0",  # Use single GPU like quantizer
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        else:
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        
        # DEBUG: Check lm_head dtype
        if hasattr(model, "lm_head"):
            print(f"DEBUG: lm_head dtype: {model.lm_head.weight.dtype}")
            if precision == "nf4" and model.lm_head.weight.dtype != torch.float32 and model.lm_head.weight.dtype != torch.bfloat16:
                print("WARNING: lm_head is not float32/bfloat16! Forcing float32...")
                # If it's quantized (Linear4bit), we can't just .float() it easily without dequantizing.
                # But if skip_modules worked, it should be a standard Linear layer.
                print(f"lm_head type: {type(model.lm_head)}")
                
                # If it is Linear4bit, then skip_modules FAILED.
                if "Linear4bit" in str(type(model.lm_head)):
                    print("CRITICAL: llm_int8_skip_modules failed! lm_head is still quantized.")
                    print("This is likely the cause of the static noise.")
        
        model.eval()
        
        # Enable PyTorch optimizations for Blackwell/Ada GPUs
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')  # Use TF32 for better throughput
        
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Try to compile the model for faster inference
        # Only compile the forward pass, not the whole generate loop
        # We check for Triton availability first
        # DISABLED: torch.compile breaks GenerationMixin attributes (_is_stateful)
        # try:
        #     import triton
        #     print("Triton detected. Attempting to compile model with torch.compile...")
        #     # Use 'max-autotune' for best performance if it works, or 'reduce-overhead' for latency
        #     # 'reduce-overhead' is often better for autoregressive generation
        #     model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        # except ImportError:
        #     print("Triton not found. Skipping torch.compile.")
        # except Exception as e:
        #     print(f"Warning: torch.compile failed: {e}")

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

        try:
            # Prefer using the imported class if available to avoid "file not found" errors for remote code
            # Use a local variable to avoid UnboundLocalError
            LocalEmu3Tokenizer = globals().get('Emu3Tokenizer')

            if LocalEmu3Tokenizer is None:
                try:
                    if current_dir not in sys.path:
                        sys.path.append(current_dir)
                    from patched_tokenization_emu3 import Emu3Tokenizer as ImportedTokenizer
                    LocalEmu3Tokenizer = ImportedTokenizer
                except ImportError as e:
                    print(f"Failed to re-import Emu3Tokenizer: {e}")

            if LocalEmu3Tokenizer is not None:
                # We construct it manually to avoid from_pretrained looking for the python file if it failed to copy
                vocab_file = os.path.join(model_path, "emu3.tiktoken")
                if os.path.exists(vocab_file):
                    print("Initializing Emu3Tokenizer directly...")
                    tokenizer = LocalEmu3Tokenizer(vocab_file=vocab_file)
                    # Load special tokens if possible, or rely on defaults
                    # We can try to load config to get special tokens
                else:
                    print(f"Warning: vocab_file not found at {vocab_file}")
                    # Fallback to from_pretrained
                    tokenizer = LocalEmu3Tokenizer.from_pretrained(
                        model_path,
                        padding_side="left"
                    )
            else:
                print("Emu3Tokenizer not in globals or is None. Using AutoTokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
        except Exception as e:
            print(f"Error loading tokenizer with preferred method: {e}")
            print("Falling back to AutoTokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
        
        # Load VQ-VAE
        # If vq_path points to a file (like model.ckpt), use its parent directory
        if os.path.isfile(vq_path):
            vq_path = os.path.dirname(vq_path)

        print(f"Loading Vision Tokenizer from {vq_path}...")
        # build_vision_tokenizer expects the directory containing model.ckpt and config.yaml
        vq_model = build_vision_tokenizer("ibq", vq_path, device=device)
        vq_model.eval()
        vq_model.to(dtype) # Match dtype if possible, or keep as is

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
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_image": ("IMAGE",), # For editing/variation
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "text_response",)
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
        
        # Calculate required tokens: H * (W + 1) for EOL tokens
        # Add some buffer for safety
        needed_tokens = latent_height * (latent_width + 1) + 50
        
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
        
        # Streamer
        # streamer = ThrottledStreamer(needed_tokens)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                # streamer=streamer
            )
            
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
        
        # Extract text response before image (everything before <|begin_of_image|>)
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

        try:
            # Extract text response (before BOI token ID)
            token_list = generated_tokens_tensor[0].tolist()
            try:
                boi_pos = token_list.index(BOI_TOKEN_ID)
                text_token_ids = token_list[:boi_pos]
                if text_token_ids:
                    text_response = tokenizer.decode(text_token_ids, skip_special_tokens=True).strip()
            except ValueError:
                # No BOI token found
                pass
            
            # Decode image from raw token IDs
            image = decode_image_from_tokens(token_list, vq_model)
            
            if image is None:
                 raise ValueError("Failed to decode image from tokens.")
                 
            # Convert PIL to Tensor (B, H, W, C) -> (B, C, H, W) for Comfy?
            # Comfy expects (B, H, W, C) in range [0, 1]
            
            import numpy as np
            from PIL import Image
            
            if isinstance(image, Image.Image):
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # Add batch dim
                return (image_tensor, text_response)
            else:
                 # If it's not a PIL image, maybe it's already a tensor or something else?
                 # But decode_image in repo returns PIL Image.
                 print(f"Warning: decode_image returned {type(image)}")
                 return (torch.zeros((1, 1024, 1024, 3)),)
            
        except Exception as e:
            print(f"Error decoding image: {e}")
            # Return a blank image or raise
            raise e

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

NODE_CLASS_MAPPINGS = {
    "Emu35Loader": Emu35Loader,
    "Emu35Sampler": Emu35Sampler,
    "Emu35VQA": Emu35VQA,
    "Emu35ClearCache": Emu35ClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35Loader": "Emu 3.5 Loader",
    "Emu35Sampler": "Emu 3.5 Sampler",
    "Emu35VQA": "Emu 3.5 VQA",
    "Emu35ClearCache": "Emu 3.5 Clear Cache",
}
