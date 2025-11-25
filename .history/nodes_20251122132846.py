import os
import sys
import torch
import folder_paths
import comfy.model_management
import comfy.utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

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
        from src.utils.generation_utils import build_logits_processor, decode_image
        # Try to import the tokenizer class directly from the repo
        try:
            from src.tokenizer_emu3_ibq.tokenization_emu3 import Emu3Tokenizer
        except ImportError:
            Emu3Tokenizer = None
            print("Could not import Emu3Tokenizer from src.tokenizer_emu3_ibq. Will rely on AutoTokenizer.")
            
    except ImportError as e:
        print(f"Error importing Emu3.5 modules: {e}")
        print("Please ensure the Emu3.5 repository is cloned into 'Emu3_5_repo' inside this node's directory.")
else:
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
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto", # Let accelerate handle it
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        else:
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map="auto", # Or manually handle device
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        
        model.eval()

        # Load Tokenizer
        print(f"Loading tokenizer from {model_path}...")
        
        # Fix for missing tokenization_emu3.py in model folder
        # Transformers requires this file to be present in the model directory for trust_remote_code=True
        try:
            import shutil
            repo_tokenizer_file = os.path.join(emu_repo_path, "src", "tokenizer_emu3_ibq", "tokenization_emu3.py")
            dest_tokenizer_file = os.path.join(model_path, "tokenization_emu3.py")
            
            if os.path.exists(repo_tokenizer_file) and not os.path.exists(dest_tokenizer_file):
                print(f"Copying tokenization_emu3.py to model folder to satisfy AutoTokenizer...")
                shutil.copy(repo_tokenizer_file, dest_tokenizer_file)
        except Exception as e:
            print(f"Warning: Could not copy tokenization_emu3.py: {e}")

        try:
            # Prefer using the imported class if available to avoid "file not found" errors for remote code
            if 'Emu3Tokenizer' in globals() and Emu3Tokenizer is not None:
                # We construct it manually to avoid from_pretrained looking for the python file if it failed to copy
                vocab_file = os.path.join(model_path, "emu3.tiktoken")
                if os.path.exists(vocab_file):
                    print("Initializing Emu3Tokenizer directly...")
                    tokenizer = Emu3Tokenizer(vocab_file=vocab_file)
                    # Load special tokens if possible, or rely on defaults
                    # We can try to load config to get special tokens
                else:
                    # Fallback to from_pretrained
                    tokenizer = Emu3Tokenizer.from_pretrained(
                        model_path,
                        padding_side="left"
                    )
            else:
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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(self, model, tokenizer, vq_model, prompt, negative_prompt, width, height, steps, cfg_scale, seed, input_image=None):
        device = model.device
        torch.manual_seed(seed)

        # Prepare Inputs
        # Emu 3.5 prompt formatting might be specific. 
        # Based on subagent, it uses standard text but might need special tokens.
        # The subagent mentioned "unconditional_ids" for CFG.
        
        # Prepare Unconditional IDs
        unconditional_ids = tokenizer.encode(negative_prompt, return_tensors="pt").to(device)
        
        # Prepare Prompt IDs
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Handle Input Image (if provided)
        # If input_image is provided, we need to encode it and prepend/insert tokens.
        # This part depends heavily on Emu 3.5's specific editing implementation.
        # For now, I'll implement basic T2I.
        # If input_image is present, we might need to use the VQ model to encode it.
        
        # Logits Processor
        # This is crucial for Emu 3.5
        logits_processor = LogitsProcessorList()
        
        class DummyConfig:
            def __init__(self):
                self.classifier_free_guidance = cfg_scale
                self.image_area = width * height
                self.h = height
                self.w = width
                self.target_height = height
                self.target_width = width
                self.image_cfg_scale = 1.0 # Default from config.py
                self.unconditional_type = "no_text" # Default from config.py
                
                # Special tokens for generation_utils
                self.special_token_ids = {
                    "PAD": tokenizer.pad_token_id,
                    "EOS": tokenizer.eos_token_id,
                    "BOS": tokenizer.bos_token_id,
                }

                # Sampling params from config.py
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
                    "max_new_tokens": steps * 10, # Will be overridden by generation_config
                    "guidance_scale": cfg_scale,
                    "use_differential_sampling": True, # Default from config.py
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
                force_same_image_size=True # or False depending on intent
            )
            logits_processor.append(processor)
        except Exception as e:
            print(f"Error building logits processor: {e}")
            # Fallback or re-raise
            raise e

        # Generation Config
        generation_config = GenerationConfig(
            max_new_tokens=5120, # Enough for 1024x1024
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=True,
            temperature=1.0, 
            top_k=10240, # Match image_top_k
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

        print("Generating...")
        
        # Create streamer for progress bar
        streamer = ComfyStreamer(generation_config.max_new_tokens)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer
            )

        # Decode
        generated_tokens = outputs[:, input_ids.shape[1]:]
        
        print("Decoding image...")
        
        # Ensure tokenizer has eol_token attribute if missing
        if not hasattr(tokenizer, "eol_token"):
            # Emu3 uses <|eol|> as end of line token
            tokenizer.eol_token = "<|eol|>"
            
        # decode_image expects a string, but we have tokens.
        # We need to decode tokens to string first?
        # Looking at generation_utils.py: decode_image(image_string, ...)
        # It splits by eol_token.
        
        # Let's try to decode the tokens to text first, as decode_image expects a string
        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
        
        try:
            image = decode_image(decoded_text, tokenizer, vq_model)
            
            if image is None:
                 raise ValueError("Failed to decode image from tokens.")
                 
            # Convert PIL to Tensor (B, H, W, C) -> (B, C, H, W) for Comfy?
            # Comfy expects (B, H, W, C) in range [0, 1]
            
            import numpy as np
            from PIL import Image
            
            if isinstance(image, Image.Image):
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # Add batch dim
                return (image_tensor,)
            else:
                 # If it's not a PIL image, maybe it's already a tensor or something else?
                 # But decode_image in repo returns PIL Image.
                 print(f"Warning: decode_image returned {type(image)}")
                 return (torch.zeros((1, 1024, 1024, 3)),)
            
        except Exception as e:
            print(f"Error decoding image: {e}")
            # Return a blank image or raise
            raise e

NODE_CLASS_MAPPINGS = {
    "Emu35Loader": Emu35Loader,
    "Emu35Sampler": Emu35Sampler,
    "Emu35VQA": Emu35VQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35Loader": "Emu 3.5 Loader",
    "Emu35Sampler": "Emu 3.5 Sampler",
    "Emu35VQA": "Emu 3.5 VQA",
}
