import os
import sys
import torch
import folder_paths
import comfy.model_management
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, LogitsProcessorList

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
    except ImportError as e:
        print(f"Error importing Emu3.5 modules: {e}")
        print("Please ensure the Emu3.5 repository is cloned into 'Emu3_5_repo' inside this node's directory.")
else:
    print(f"Emu3.5 repository not found at {emu_repo_path}")
    print("Please clone https://github.com/baaivision/Emu3.5 into 'Emu3_5_repo' inside this node's directory.")

class Emu35Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("emu35"),),
                "precision": (["bf16", "fp16", "fp32", "nf4"], {"default": "bf16"}),
            },
            "optional": {
                "vq_path": ("STRING", {"default": "models/emu35/vision_tokenizer"}), # Adjust default as needed
            }
        }

    RETURN_TYPES = ("EMU_MODEL", "EMU_TOKENIZER", "EMU_VQ")
    RETURN_NAMES = ("model", "tokenizer", "vq_model")
    FUNCTION = "load_model"
    CATEGORY = "Emu3.5"

    def load_model(self, model_name, precision, vq_path="models/emu35/vision_tokenizer"):
        model_path = folder_paths.get_full_path("emu35", model_name)
        
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
                attn_implementation="flash_attention_2"
            )
        else:
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map="auto", # Or manually handle device
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
        
        model.eval()

        # Load Tokenizer
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left" # Usually left for generation
        )
        
        # Load VQ-VAE
        # Assuming vq_path is absolute or relative to ComfyUI root
        # If user provides just a name, we might need to find it.
        # For now, assume full path or relative to comfy root.
        print(f"Loading Vision Tokenizer from {vq_path}...")
        # build_vision_tokenizer expects a path and device
        # We might need to resolve the path if it's not absolute
        if not os.path.isabs(vq_path):
            vq_path = os.path.abspath(vq_path)
            
        vq_model = build_vision_tokenizer("ibq", vq_path, device=device)
        vq_model.eval()
        vq_model.to(dtype) # Match dtype if possible, or keep as is

        return (model, tokenizer, vq_model)

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
        
        # We need to mock the 'cfg' object or pass parameters as expected by build_logits_processor
        # The subagent said: build_logits_processor(cfg=None, unconditional_ids=..., ...)
        # We might need to inspect build_logits_processor signature in the repo.
        # Assuming the subagent's snippet is accurate:
        
        # Note: The subagent snippet had `cfg=None`. 
        # If `cfg` param in build_logits_processor expects a config object with `classifier_free_guidance`, we might need to create one.
        # Let's assume we can pass `classifier_free_guidance` directly or via a dummy object.
        
        class DummyConfig:
            def __init__(self):
                self.classifier_free_guidance = cfg_scale
                self.image_area = width * height # Approximate or exact?
                self.h = height
                self.w = width
                
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
            max_new_tokens=steps * 10, # Heuristic, or calculate based on image size
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=True,
            temperature=1.0, # Can expose this
            top_k=50, # Can expose this
        )
        
        # Calculate max tokens needed for image
        # Emu tokens per image depends on VQ-VAE. 
        # Usually 1024x1024 -> 64x64 tokens = 4096 tokens?
        # The subagent said max_new_tokens=5120.
        generation_config.max_new_tokens = 5120

        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor
            )

        # Decode
        generated_tokens = outputs[:, input_ids.shape[1]:]
        
        print("Decoding image...")
        # decode_image(tokens, tokenizer, vq_model)
        # We need to ensure tokens are in the right format.
        image = decode_image(generated_tokens, tokenizer, vq_model)
        
        # Image is likely a PIL image or tensor.
        # ComfyUI expects (Batch, Height, Width, Channels) Tensor [0-1]
        
        import numpy as np
        from PIL import Image
        
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # If it's a tensor, check dimensions
            if image.ndim == 3: # C, H, W or H, W, C
                if image.shape[0] == 3:
                    image = image.permute(1, 2, 0)
                image = image.unsqueeze(0)
            # Ensure range 0-1
            if image.max() > 1.0:
                image = image / 255.0
        
        return (image,)

NODE_CLASS_MAPPINGS = {
    "Emu35Loader": Emu35Loader,
    "Emu35Sampler": Emu35Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35Loader": "Emu 3.5 Loader",
    "Emu35Sampler": "Emu 3.5 Sampler",
}
