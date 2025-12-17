"""
Emu3.5 ComfyUI Nodes - Version 2.0
Optimized for speed and memory with full feature support.

Features:
- T2I (Text-to-Image)
- X2I (Any-to-Image / Image Editing with reference images)
- Interleaved Generation (Story/HowTo)
- VQA (Visual Question Answering)
- Tiled VQ Decoding (reduced VRAM for decode)
- Smart memory management
"""

import os
import sys
import gc
import math
import re
import time
from typing import List, Optional, Tuple, Dict, Any, Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management
import comfy.utils

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    GenerationConfig, 
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList
)
from transformers.generation.streamers import BaseStreamer

# =============================================================================
# Setup paths and imports
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")

# Register emu35 model folder
try:
    folder_paths.add_model_folder_path("emu35", os.path.join(folder_paths.models_dir, "emu35"))
except:
    pass

# Import Emu3.5 modules
if os.path.exists(emu_repo_path):
    sys.path.insert(0, emu_repo_path)
    try:
        from src.emu3p5 import Emu3ForCausalLM, Emu3Config
        from src.vision_tokenizer import build_vision_tokenizer
        from src.utils.generation_utils import build_logits_processor, decode_image, multimodal_decode
        from src.utils.input_utils import smart_resize, format_image_string, build_image
        EMU_AVAILABLE = True
    except ImportError as e:
        print(f"Error importing Emu3.5 modules: {e}")
        EMU_AVAILABLE = False
else:
    print(f"Emu3.5 repository not found at {emu_repo_path}")
    EMU_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================

# Token IDs
BOS_TOKEN_ID = 151849  # <|extra_203|>
EOS_TOKEN_ID = 151850  # <|extra_204|>
PAD_TOKEN_ID = 151643  # <|endoftext|>
IMG_TOKEN_ID = 151851  # <|image token|>
BOI_TOKEN_ID = 151852  # <|image start|>
EOI_TOKEN_ID = 151853  # <|image end|>
EOL_TOKEN_ID = 151846  # <|extra_200|>
BOV_TOKEN_ID = 151854  # First visual token

# Aspect ratio presets (latent dimensions)
ASPECT_RATIOS = {
    "1:1": (64, 64),      # 1024x1024
    "4:3": (55, 73),      # 880x1168
    "3:4": (73, 55),      # 1168x880
    "16:9": (47, 85),     # 752x1360
    "9:16": (85, 47),     # 1360x752
    "3:2": (52, 78),      # 832x1248
    "2:3": (78, 52),      # 1248x832
    "21:9": (41, 97),     # 656x1552
    "9:21": (97, 41),     # 1552x656
}

# Task templates
TASK_TEMPLATES = {
    "t2i": {
        "with_image": "<|extra_203|>You are a helpful assistant for t2i task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>",
        "without_image": "<|extra_203|>You are a helpful assistant for t2i task. USER: {question} ASSISTANT: <|extra_100|>",
        "unconditional": "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>",
    },
    "x2i": {
        "with_image": "<|extra_203|>You are a helpful assistant for x2i task. USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>",
        "without_image": "<|extra_203|>You are a helpful assistant for x2i task. USER: {question} ASSISTANT: <|extra_100|>",
        "unconditional": "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>",
    },
    "story": {
        "with_image": "<|extra_203|>You are a helpful assistant for story task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>",
        "without_image": "<|extra_203|>You are a helpful assistant for story task. USER: {question} ASSISTANT: <|extra_100|>",
        "unconditional": "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>",
    },
    "howto": {
        "with_image": "<|extra_203|>You are a helpful assistant for howto task. Please generate a response with interleaved text and images. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>",
        "without_image": "<|extra_203|>You are a helpful assistant for howto task. Please generate a response with interleaved text and images. USER: {question} ASSISTANT: <|extra_100|>",
        "unconditional": "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>",
    },
}

# Default sampling parameters by task
SAMPLING_PRESETS = {
    "t2i": {
        "text_top_k": 1024,
        "text_top_p": 0.9,
        "text_temperature": 1.0,
        "image_top_k": 5120,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "cfg_scale": 5.0,
        "max_new_tokens": 5120,
    },
    "x2i": {
        "text_top_k": 1024,
        "text_top_p": 0.9,
        "text_temperature": 1.0,
        "image_top_k": 5120,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "cfg_scale": 3.0,
        "max_new_tokens": 5120,
    },
    "story": {
        "text_top_k": 1024,
        "text_top_p": 0.9,
        "text_temperature": 1.0,
        "image_top_k": 10240,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "cfg_scale": 3.0,
        "max_new_tokens": 32768,
    },
    "howto": {
        "text_top_k": 200,
        "text_top_p": 0.8,
        "text_temperature": 0.7,
        "image_top_k": 10240,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "cfg_scale": 3.0,
        "max_new_tokens": 32768,
    },
}

# =============================================================================
# Utility Classes
# =============================================================================

class ProgressStreamer(BaseStreamer):
    """Streamer that updates ComfyUI progress bar."""
    def __init__(self, total_steps: int, update_every: int = 32):
        self.pbar = comfy.utils.ProgressBar(total_steps)
        self.counter = 0
        self.total_steps = total_steps
        self.update_every = update_every

    def put(self, value):
        self.counter += 1
        if self.counter % self.update_every == 0 or self.counter == self.total_steps:
            self.pbar.update(min(self.update_every, self.counter))
        return value

    def end(self):
        pass


class StopAfterImage(StoppingCriteria):
    """Stop generation after seeing a NEW EOI token (not one from input)."""
    def __init__(self, eoi_token_id: int = EOI_TOKEN_ID, input_length: int = 0):
        self.eoi_token_id = eoi_token_id
        self.input_length = input_length  # Length of original input (to ignore EOI in input)
        self.found_eoi = False

    def __call__(self, input_ids, scores, **kwargs):
        if self.found_eoi:
            return True
        # Only check tokens AFTER the original input
        generated_tokens = input_ids[0, self.input_length:]
        if (generated_tokens == self.eoi_token_id).any().item():
            self.found_eoi = True
            return True
        return False


class Emu35Config:
    """Configuration object for Emu3.5 generation."""
    def __init__(
        self,
        task_type: str = "t2i",
        cfg_scale: float = 5.0,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        text_top_k: int = 1024,
        text_top_p: float = 0.9,
        text_temperature: float = 1.0,
        image_top_k: int = 5120,
        image_top_p: float = 1.0,
        image_temperature: float = 1.0,
        max_new_tokens: int = 5120,
        image_area: int = 1048576,
    ):
        self.classifier_free_guidance = cfg_scale
        self.unconditional_type = "no_text"
        self.image_cfg_scale = 1.0
        self.target_height = target_height
        self.target_width = target_width
        self.image_area = image_area
        
        self.sampling_params = {
            "use_cache": True,
            "text_top_k": text_top_k,
            "text_top_p": text_top_p,
            "text_temperature": text_temperature,
            "image_top_k": image_top_k,
            "image_top_p": image_top_p,
            "image_temperature": image_temperature,
            "top_k": 131072,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams_per_group": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "max_new_tokens": max_new_tokens,
            "guidance_scale": 1.0,
            "use_differential_sampling": True,
            "do_sample": True,
            "num_beams": 1,
        }


# =============================================================================
# Memory Management Utilities  
# =============================================================================

@contextmanager
def torch_gc():
    """Context manager for aggressive garbage collection."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_model_device(model) -> torch.device:
    """Get the device of a model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_model_to_device(model, device: str, dtype: Optional[torch.dtype] = None):
    """Safely move model to device."""
    if dtype:
        model.to(device=device, dtype=dtype)
    else:
        model.to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Tiled VQ Decoding
# =============================================================================

def decode_image_tiled(
    vq_model,
    code: torch.Tensor,
    tile_size: int = 32,
    overlap: int = 4,
) -> Image.Image:
    """
    Decode VQ codes using tiled processing to reduce peak VRAM usage.
    
    Args:
        vq_model: Vision tokenizer model
        code: Tensor of shape [H, W] containing VQ code indices
        tile_size: Size of each tile in latent space
        overlap: Overlap between tiles for blending
    
    Returns:
        PIL Image
    """
    device = next(vq_model.parameters()).device
    h, w = code.shape
    
    # If image is small enough, decode directly
    if h <= tile_size and w <= tile_size:
        image = vq_model.decode_code(code[None], shape=(1, h, w, 256)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(
            ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        return image
    
    # Calculate output size (16x upscale)
    out_h, out_w = h * 16, w * 16
    tile_out_size = tile_size * 16
    overlap_out = overlap * 16
    
    # Create output tensor on CPU to save VRAM
    output = torch.zeros((3, out_h, out_w), dtype=torch.float32, device="cpu")
    weight = torch.zeros((1, out_h, out_w), dtype=torch.float32, device="cpu")
    
    # Process tiles
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Calculate tile bounds
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile_h = y_end - y
            tile_w = x_end - x
            
            # Extract tile
            tile_code = code[y:y_end, x:x_end].to(device)
            
            # Decode tile
            with torch.no_grad():
                tile_img = vq_model.decode_code(
                    tile_code[None], 
                    shape=(1, tile_h, tile_w, 256)
                ).float()
            
            # Move to CPU immediately to free VRAM
            tile_img = tile_img[0].cpu()  # [C, H, W]
            
            # Calculate output position
            out_y = y * 16
            out_x = x * 16
            out_y_end = y_end * 16
            out_x_end = x_end * 16
            
            # Create blending weight (feather edges for overlap regions)
            tile_weight = torch.ones((1, tile_h * 16, tile_w * 16), dtype=torch.float32)
            
            # Feather edges if not at boundary
            if overlap > 0:
                feather = overlap_out
                # Top edge
                if y > 0:
                    for i in range(feather):
                        tile_weight[:, i, :] *= i / feather
                # Bottom edge
                if y_end < h:
                    for i in range(feather):
                        tile_weight[:, -(i+1), :] *= i / feather
                # Left edge
                if x > 0:
                    for i in range(feather):
                        tile_weight[:, :, i] *= i / feather
                # Right edge
                if x_end < w:
                    for i in range(feather):
                        tile_weight[:, :, -(i+1)] *= i / feather
            
            # Accumulate
            output[:, out_y:out_y_end, out_x:out_x_end] += tile_img * tile_weight
            weight[:, out_y:out_y_end, out_x:out_x_end] += tile_weight
            
            # Free VRAM
            del tile_code, tile_img
            torch.cuda.empty_cache()
    
    # Normalize by weight
    output = output / weight.clamp(min=1e-6)
    
    # Convert to PIL
    output = output.permute(1, 2, 0)  # [H, W, C]
    image = Image.fromarray(
        ((output + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    )
    
    return image


# =============================================================================
# Helper Functions
# =============================================================================

def get_emu_subfolders() -> List[str]:
    """List subdirectories in the emu35 model folder."""
    model_paths = folder_paths.get_folder_paths("emu35")
    subfolders = []
    for path in model_paths:
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    subfolders.append(item)
    return sorted(list(set(subfolders)))


def find_folder_path(folder_name: str) -> Optional[str]:
    """Find full path for a folder in emu35 model paths."""
    for path in folder_paths.get_folder_paths("emu35"):
        full_path = os.path.join(path, folder_name)
        if os.path.exists(full_path):
            return full_path
    return None


def setup_tokenizer_attributes(tokenizer):
    """Set required special token attributes on tokenizer."""
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


def encode_reference_images(
    images: List[torch.Tensor],
    vq_model,
    tokenizer,
    image_area: int = 1048576,
) -> str:
    """
    Encode reference images to token strings for X2I tasks.
    
    Args:
        images: List of ComfyUI image tensors [B, H, W, C]
        vq_model: Vision tokenizer
        tokenizer: Text tokenizer
        image_area: Target image area for resizing
    
    Returns:
        Combined image token string
    """
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    
    image_strings = []
    
    for idx, img_tensor in enumerate(images):
        # Take first image in batch
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        
        # Convert to PIL for smart_resize
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        print(f"[Encode] Image {idx+1}: Original size {pil_img.size}")
        
        # Resize to target area
        pil_img = smart_resize(pil_img, image_area)
        w, h = pil_img.size
        
        print(f"[Encode] Image {idx+1}: Resized to {w}x{h} ({h//16}x{w//16} latents)")
        
        # Convert to tensor [-1, 1]
        img_input = torch.tensor((np.array(pil_img) / 127.5 - 1.0), dtype=dtype, device=device)
        img_input = img_input.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Encode
        with torch.no_grad():
            _, _, token = vq_model.encode(img_input)
        
        # Reshape tokens
        token = token[-1].view(h // 16, w // 16)
        
        print(f"[Encode] Image {idx+1}: Token shape {token.shape}, range [{token.min()}, {token.max()}]")
        
        # Format to string
        img_str = format_image_string(tokenizer, token)
        
        # Verify the string format
        if tokenizer.boi_token in img_str and tokenizer.eoi_token in img_str:
            print(f"[Encode] Image {idx+1}: Token string OK ({len(img_str)} chars)")
        else:
            print(f"[Encode] Image {idx+1}: WARNING - Missing BOI/EOI tokens!")
            print(f"[Encode] String start: {img_str[:100]}")
        
        image_strings.append(img_str)
    
    combined = "".join(image_strings)
    print(f"[Encode] Total combined string: {len(combined)} chars")
    return combined


def comfy_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    if image.dim() == 4:
        image = image[0]
    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_comfy(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor."""
    img_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


# =============================================================================
# Node: Emu35 Loader (Improved)
# =============================================================================

class Emu35LoaderV2:
    """
    Load Emu3.5 model with improved memory management.
    Supports both Emu3.5-Image and Emu3.5 base models.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        available_folders = get_emu_subfolders()
        if not available_folders:
            available_folders = ["No folders found in models/emu35"]
        
        vq_default = "vision_tokenizer" if "vision_tokenizer" in available_folders else available_folders[0]
        
        return {
            "required": {
                "model_name": (available_folders,),
                "vq_model_name": (available_folders, {"default": vq_default}),
                "precision": (["auto", "bf16", "fp16", "fp32", "nf4 (quantize)"], {"default": "auto"}),
                "device": (["cuda:0", "cuda:1", "auto"], {"default": "cuda:0"}),
            },
            "optional": {
                "vq_device": (["same", "cuda:0", "cuda:1", "cpu"], {"default": "same"}),
            }
        }

    RETURN_TYPES = ("EMU_MODEL", "EMU_TOKENIZER", "EMU_VQ", "EMU_DEVICE_INFO")
    RETURN_NAMES = ("model", "tokenizer", "vq_model", "device_info")
    FUNCTION = "load_model"
    CATEGORY = "Emu3.5"

    def load_model(self, model_name, vq_model_name, precision, device, vq_device="same"):
        import json
        
        if not EMU_AVAILABLE:
            raise RuntimeError("Emu3.5 modules not available. Check installation.")
        
        model_path = find_folder_path(model_name)
        vq_path = find_folder_path(vq_model_name)
        
        if model_path is None:
            raise ValueError(f"Could not find model folder: {model_name}")
        if vq_path is None:
            raise ValueError(f"Could not find VQ folder: {vq_model_name}")

        # Determine device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if vq_device == "same":
            vq_device = device

        # Check for pre-quantized model
        config_path = os.path.join(model_path, "config.json")
        is_pre_quantized = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_json = json.load(f)
                if "quantization_config" in config_json:
                    is_pre_quantized = True
                    print(f"✓ Detected pre-quantized model")

        # Determine loading strategy
        if precision == "auto":
            load_mode = "pre_quantized" if is_pre_quantized else "bf16"
            dtype = torch.bfloat16
        elif precision == "nf4 (quantize)":
            load_mode = "pre_quantized" if is_pre_quantized else "quantize_nf4"
            dtype = torch.bfloat16
        else:
            load_mode = precision
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        print(f"Loading Emu3.5 model: {model_name}")
        print(f"  Device: {device}, VQ Device: {vq_device}")
        print(f"  Precision: {precision} -> {load_mode}")

        # Load Config
        try:
            config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
            from transformers import AutoConfig
            AutoConfig.register("Emu3", Emu3Config)
        except Exception:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Use eager attention (sdpa has issues on Blackwell)
        attn_impl = "eager"

        # Load Model
        if load_mode == "pre_quantized":
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        elif load_mode == "quantize_nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head", "model.embed_tokens", "model.norm"]
            )
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        else:
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
        
        model.eval()
        
        # Enable optimizations
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load Tokenizer
        print(f"Loading tokenizer...")
        
        # Copy patched tokenizer
        try:
            import shutil
            patched_file = os.path.join(current_dir, "patched_tokenization_emu3.py")
            dest_file = os.path.join(model_path, "tokenization_emu3.py")
            if os.path.exists(patched_file):
                shutil.copy(patched_file, dest_file)
        except Exception as e:
            print(f"Warning: Could not copy tokenizer: {e}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer = setup_tokenizer_attributes(tokenizer)
        
        # Ensure model has generation_config
        if model.generation_config is None:
            model.generation_config = GenerationConfig(
                bos_token_id=BOS_TOKEN_ID,
                eos_token_id=EOS_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
            )

        # Load VQ-VAE
        print(f"Loading Vision Tokenizer on {vq_device}...")
        vq_model = build_vision_tokenizer("ibq", vq_path, device=vq_device)
        vq_model.eval()

        # Device info for other nodes
        device_info = {
            "model_device": device,
            "vq_device": vq_device,
            "dtype": str(dtype),
            "model_name": model_name,
        }

        print(f"✓ Model loaded successfully")
        return (model, tokenizer, vq_model, device_info)


# =============================================================================
# Node: Emu35 T2I Sampler (Improved)
# =============================================================================

class Emu35T2ISamplerV2:
    """
    Text-to-Image generation with full control over sampling parameters.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over mountains"}),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9", "9:21"],),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image_top_k": ("INT", {"default": 5120, "min": 100, "max": 131072, "step": 100}),
                "image_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "text_top_k": ("INT", {"default": 1024, "min": 100, "max": 10000, "step": 100}),
                "text_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 32, "min": 16, "max": 64, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "text_response", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(
        self, 
        model, 
        tokenizer, 
        vq_model, 
        prompt, 
        aspect_ratio, 
        cfg_scale, 
        seed,
        image_top_k=5120,
        image_temperature=1.0,
        text_top_k=1024,
        text_temperature=1.0,
        tiled_decode=False,
        tile_size=32,
    ):
        torch.manual_seed(seed)
        device = get_model_device(model)
        
        # Get target size
        target_height, target_width = ASPECT_RATIOS[aspect_ratio]
        
        print(f"[T2I] Generating {target_height*16}x{target_width*16} image")
        print(f"[T2I] CFG: {cfg_scale}, Seed: {seed}")
        
        # Build config
        cfg = Emu35Config(
            task_type="t2i",
            cfg_scale=cfg_scale,
            target_height=target_height,
            target_width=target_width,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            image_top_k=image_top_k,
            image_temperature=image_temperature,
            max_new_tokens=5120,
        )
        
        # Build prompts
        template = TASK_TEMPLATES["t2i"]["without_image"]
        unc_prompt = TASK_TEMPLATES["t2i"]["unconditional"]
        
        full_prompt = template.format(question=prompt)
        
        # Encode
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Build logits processor
        logits_processor = LogitsProcessorList()
        logits_processor.append(
            build_logits_processor(
                cfg=cfg,
                unconditional_ids=unconditional_ids,
                model=model,
                tokenizer=tokenizer,
                force_same_image_size=True,
            )
        )
        
        # Build generation config
        generation_config = GenerationConfig(
            **cfg.sampling_params,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        
        # Stopping criteria - pass input length to ignore any EOI in input
        stopping_criteria = StoppingCriteriaList([StopAfterImage(input_length=input_ids.shape[1])])
        
        # Progress tracking
        expected_tokens = target_height * (target_width + 1) + 100
        streamer = ProgressStreamer(expected_tokens)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
        
        gen_time = time.time() - start_time
        print(f"[T2I] Generation took {gen_time:.1f}s ({outputs.shape[1]/gen_time:.1f} tok/s)")
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Use custom tiled decode if enabled
        if tiled_decode:
            image, text_response, reasoning = self._decode_with_tiles(
                output_text, tokenizer, vq_model, tile_size
            )
        else:
            image, text_response, reasoning = self._decode_standard(
                output_text, tokenizer, vq_model
            )
        
        return (image, text_response, reasoning)
    
    def _decode_standard(self, output_text, tokenizer, vq_model):
        """Standard decoding using official multimodal_decode."""
        try:
            mm_output = multimodal_decode(output_text, tokenizer, vq_model)
            
            image = None
            text_response = ""
            reasoning = ""
            
            for item_type, item_data in mm_output:
                if item_type == "image" and item_data is not None:
                    image = pil_to_comfy(item_data)
                elif item_type == "text":
                    text_response += item_data
                elif item_type == "global_cot":
                    reasoning += f"[Global] {item_data}\n"
                elif item_type == "image_cot":
                    reasoning += f"[Image] {item_data}\n"
            
            if image is not None:
                return (image, text_response, reasoning)
            
        except Exception as e:
            print(f"[T2I] Decode error: {e}")
            import traceback
            traceback.print_exc()
        
        # Return empty on failure
        return (torch.zeros(1, 512, 512, 3), "", "")
    
    def _decode_with_tiles(self, output_text, tokenizer, vq_model, tile_size):
        """Tiled decoding for reduced VRAM."""
        # Extract image tokens manually
        pattern = rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)}"
        matches = re.findall(pattern, output_text, re.DOTALL)
        
        if not matches:
            return (torch.zeros(1, 512, 512, 3), "", "No image found")
        
        image_string = matches[0]
        
        # Parse visual tokens
        image_tokens: List[List[int]] = []
        image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
        
        for r in image_rows:
            token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
            if token_ids:
                row_tokens = [int(m) for m in token_ids]
                image_tokens.append(row_tokens)
        
        if not image_tokens:
            return (torch.zeros(1, 512, 512, 3), "", "No visual tokens")
        
        # Convert to tensor
        device = next(vq_model.parameters()).device
        code = torch.tensor(image_tokens, dtype=torch.long, device=device)
        
        # Decode with tiles
        pil_image = decode_image_tiled(vq_model, code, tile_size=tile_size, overlap=4)
        
        # Extract text
        text_parts = re.split(pattern, output_text, flags=re.DOTALL)
        text_response = " ".join(t.strip() for t in text_parts if t.strip())
        
        return (pil_to_comfy(pil_image), text_response, "")


# =============================================================================
# Node: Emu35 X2I Sampler (Image Editing)
# =============================================================================

class Emu35X2ISampler:
    """
    Any-to-Image generation - edit or transform images based on text + reference images.
    Supports 1-3 reference images.
    
    Tips:
    - Use Emu3.5-Image model for best quality
    - CFG 2-3 works well for X2I
    - Describe what you want clearly, referencing "the image" or "first/second image"
    - Use smaller image_area (e.g. 262144 = 512x512) if you get OOM errors
    """
    
    # Image area presets (determines reference image encoding size)
    IMAGE_AREA_PRESETS = {
        "256x256 (65536) - Fastest": 65536,
        "384x384 (147456)": 147456,
        "512x512 (262144) - Recommended": 262144,
        "640x640 (409600)": 409600,
        "768x768 (589824)": 589824,
        "1024x1024 (1048576) - High VRAM": 1048576,
    }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "prompt": ("STRING", {"multiline": True, "default": "Transform this image into a watercolor painting style"}),
                "reference_image_1": ("IMAGE",),
                "image_area": (list(s.IMAGE_AREA_PRESETS.keys()), {"default": "512x512 (262144) - Recommended"}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "image_top_k": ("INT", {"default": 5120, "min": 100, "max": 131072, "step": 100}),
                "image_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 32, "min": 16, "max": 64, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "text_response", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(
        self,
        model,
        tokenizer,
        vq_model,
        prompt,
        reference_image_1,
        image_area,
        cfg_scale,
        seed,
        reference_image_2=None,
        reference_image_3=None,
        image_top_k=5120,
        image_temperature=1.0,
        tiled_decode=False,
        tile_size=32,
    ):
        torch.manual_seed(seed)
        device = get_model_device(model)
        
        # Validate prompt
        if not prompt or len(prompt.strip()) < 10:
            print(f"[X2I] WARNING: Prompt is very short: '{prompt}'")
            print(f"[X2I] TIP: X2I needs a detailed edit instruction like:")
            print(f"[X2I]   - 'Change the background to a sunset'")
            print(f"[X2I]   - 'Add sunglasses to the person'")
            print(f"[X2I]   - 'Replace the dog with a cat'")
            print(f"[X2I]   For 2+ images: 'Replace the [object] in first image with [object] from second image'")
        
        # Get actual image area value from preset name
        actual_image_area = self.IMAGE_AREA_PRESETS.get(image_area, 262144)
        
        # Collect reference images
        ref_images = [reference_image_1]
        if reference_image_2 is not None:
            ref_images.append(reference_image_2)
        if reference_image_3 is not None:
            ref_images.append(reference_image_3)
        
        num_refs = len(ref_images)
        print(f"[X2I] Using {num_refs} reference image(s)")
        print(f"[X2I] Image area: {actual_image_area} ({int(actual_image_area**0.5)}x{int(actual_image_area**0.5)} approx)")
        
        # Estimate memory requirement
        # Each image at area A produces (A/256) tokens approximately
        estimated_tokens = int(num_refs * actual_image_area / 256) + 100
        print(f"[X2I] Estimated input tokens: ~{estimated_tokens}")
        if estimated_tokens > 4000:
            print(f"[X2I] WARNING: High token count may cause OOM. Consider using smaller image_area.")
        
        # Encode reference images
        print("[X2I] Encoding reference images...")
        try:
            image_token_string = encode_reference_images(
                ref_images, vq_model, tokenizer, image_area=actual_image_area
            )
        except Exception as e:
            print(f"[X2I] ERROR encoding reference images: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros(1, 512, 512, 3), "", f"Error encoding images: {e}")
        
        if not image_token_string or len(image_token_string) < 50:
            print(f"[X2I] ERROR: Failed to encode reference images (token string too short)")
            return (torch.zeros(1, 512, 512, 3), "", "Failed to encode reference images")
        
        print(f"[X2I] Image tokens length: {len(image_token_string)} chars")
        
        # Build config
        # For X2I, we don't set target size - let model decide based on input
        cfg = Emu35Config(
            task_type="x2i",
            cfg_scale=cfg_scale,
            target_height=None,  # Auto-determine from generation
            target_width=None,
            image_top_k=image_top_k,
            image_temperature=image_temperature,
            max_new_tokens=5120,
        )
        
        # Build prompt with image tokens - OFFICIAL template: <|IMAGE|>{question}
        # Image tokens come BEFORE the question text
        template = TASK_TEMPLATES["x2i"]["with_image"]
        full_prompt = template.format(question=prompt)
        
        print(f"[X2I] User prompt: '{prompt}'")
        print(f"[X2I] Template (before image): {template[:50]}...{template[-50:]}")
        
        # Debug: Check image token string
        if not image_token_string:
            print(f"[X2I] ERROR: image_token_string is empty!")
        elif len(image_token_string) < 100:
            print(f"[X2I] WARNING: image_token_string seems too short: {len(image_token_string)} chars")
            print(f"[X2I] image_token_string: {image_token_string}")
        
        # Check template has placeholder
        if "<|IMAGE|>" not in full_prompt:
            print(f"[X2I] ERROR: Template doesn't contain <|IMAGE|> placeholder!")
            print(f"[X2I] Template: {template}")
            print(f"[X2I] Full prompt: {full_prompt[:200]}")
        
        # Replace placeholder
        full_prompt = full_prompt.replace("<|IMAGE|>", image_token_string)
        
        # Verify replacement worked
        if "<|IMAGE|>" in full_prompt:
            print(f"[X2I] ERROR: <|IMAGE|> still in prompt after replacement!")
        else:
            print(f"[X2I] Prompt built successfully, length: {len(full_prompt)} chars")
        
        # Unconditional also needs image tokens for proper CFG
        unc_template = TASK_TEMPLATES["x2i"]["unconditional"]
        unc_prompt = unc_template.replace("<|IMAGE|>", image_token_string)
        
        # Encode prompts
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # CRITICAL: Add BOS token if not present (official code does this)
        bos_id = BOS_TOKEN_ID
        if input_ids[0, 0] != bos_id:
            bos_tensor = torch.tensor([[bos_id]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([bos_tensor, input_ids], dim=1)
            print(f"[X2I] Added BOS token")
        
        if unconditional_ids[0, 0] != bos_id:
            bos_tensor = torch.tensor([[bos_id]], device=device, dtype=unconditional_ids.dtype)
            unconditional_ids = torch.cat([bos_tensor, unconditional_ids], dim=1)
        
        print(f"[X2I] Input tokens: {input_ids.shape[1]}, Unconditional tokens: {unconditional_ids.shape[1]}")
        
        # For multiple reference images, don't force same size
        force_same_size = (num_refs == 1)
        
        # Build logits processor
        logits_processor = LogitsProcessorList()
        logits_processor.append(
            build_logits_processor(
                cfg=cfg,
                unconditional_ids=unconditional_ids,
                model=model,
                tokenizer=tokenizer,
                force_same_image_size=force_same_size,
            )
        )
        
        # Generate
        generation_config = GenerationConfig(
            **cfg.sampling_params,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        
        # CRITICAL: Pass input_length to stop criteria - input already has EOI tokens from reference images!
        stopping_criteria = StoppingCriteriaList([StopAfterImage(input_length=input_ids.shape[1])])
        streamer = ProgressStreamer(5000)
        
        print(f"[X2I] Starting generation with CFG={cfg_scale}...")
        print(f"[X2I] Generation config: max_new_tokens={cfg.sampling_params['max_new_tokens']}, do_sample={cfg.sampling_params['do_sample']}")
        print(f"[X2I] Image sampling: top_k={cfg.sampling_params['image_top_k']}, top_p={cfg.sampling_params['image_top_p']}, temp={cfg.sampling_params['image_temperature']}")
        
        # Clear CUDA cache to maximize available VRAM for generation
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[X2I] CUDA cache cleared. Available VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")
        
        start_time = time.time()
        
        print(f"[X2I] About to call model.generate()...")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    streamer=streamer,
                )
        except Exception as e:
            print(f"[X2I] ERROR during generation: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        gen_time = time.time() - start_time
        gen_tokens = outputs.shape[1] - input_ids.shape[1]
        print(f"[X2I] Generation took {gen_time:.1f}s, generated {gen_tokens} tokens")
        
        # Check if we actually generated new tokens
        if gen_tokens < 10:
            print(f"[X2I] WARNING: Very few tokens generated ({gen_tokens}). Model may not have understood the task.")
            # Debug: show what was generated
            new_tokens = outputs[0, input_ids.shape[1]:]
            if len(new_tokens) > 0:
                debug_text = tokenizer.decode(new_tokens[:50], skip_special_tokens=False)
                print(f"[X2I] Generated start: {debug_text[:200]}")
                print(f"[X2I] Generated token IDs: {new_tokens.tolist()}")
                print(f"[X2I] EOS token ID: {EOS_TOKEN_ID}, BOI: {BOI_TOKEN_ID}, EOI: {EOI_TOKEN_ID}")
                # Check last few input tokens to see context
                last_input = input_ids[0, -10:].tolist()
                print(f"[X2I] Last 10 input tokens: {last_input}")
                print(f"[X2I] Last 10 decoded: {tokenizer.decode(last_input)}")
        
        # Decode - only decode the GENERATED part, not the input
        # This avoids re-decoding the input reference images
        generated_ids = outputs[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        print(f"[X2I] Decoded output length: {len(output_text)} chars")
        
        # Check if output contains image tokens
        if tokenizer.boi_token in output_text and tokenizer.eoi_token in output_text:
            print(f"[X2I] Found image tokens in output")
        else:
            print(f"[X2I] WARNING: No complete image found in output")
            print(f"[X2I] Output preview: {output_text[:500]}")
        
        # Decode image
        if tiled_decode:
            image, text_response, reasoning = self._decode_with_tiles(
                output_text, tokenizer, vq_model, tile_size
            )
        else:
            image, text_response, reasoning = self._decode_standard(
                output_text, tokenizer, vq_model
            )
        
        return (image, text_response, reasoning)
    
    def _decode_standard(self, output_text, tokenizer, vq_model):
        """Standard decoding using official multimodal_decode."""
        try:
            mm_output = multimodal_decode(output_text, tokenizer, vq_model)
            
            image = None
            text_response = ""
            reasoning = ""
            
            for item_type, item_data in mm_output:
                if item_type == "image" and item_data is not None:
                    # Take the LAST image (in case there are multiple)
                    image = pil_to_comfy(item_data)
                elif item_type == "text":
                    text_response += item_data
                elif item_type == "global_cot":
                    reasoning += f"[Global] {item_data}\n"
                elif item_type == "image_cot":
                    reasoning += f"[Image] {item_data}\n"
            
            if image is not None:
                return (image, text_response, reasoning)
            else:
                print("[X2I] No image decoded from output")
                
        except Exception as e:
            print(f"[X2I] Decode error: {e}")
            import traceback
            traceback.print_exc()
        
        return (torch.zeros(1, 512, 512, 3), "", "No image generated")
    
    def _decode_with_tiles(self, output_text, tokenizer, vq_model, tile_size):
        """Tiled decoding for reduced VRAM."""
        # Extract image tokens manually
        pattern = rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)}"
        matches = re.findall(pattern, output_text, re.DOTALL)
        
        if not matches:
            print("[X2I] No image block found in output for tiled decode")
            return (torch.zeros(1, 512, 512, 3), "", "No image found")
        
        # Take the last image block (the generated output)
        image_string = matches[-1]
        
        # Parse visual tokens
        image_tokens: List[List[int]] = []
        image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
        
        for r in image_rows:
            token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
            if token_ids:
                row_tokens = [int(m) for m in token_ids]
                image_tokens.append(row_tokens)
        
        if not image_tokens:
            return (torch.zeros(1, 512, 512, 3), "", "No visual tokens")
        
        print(f"[X2I] Tiled decode: {len(image_tokens)} rows x {len(image_tokens[0])} cols")
        
        # Convert to tensor
        vq_device = next(vq_model.parameters()).device
        code = torch.tensor(image_tokens, dtype=torch.long, device=vq_device)
        
        # Decode with tiles
        pil_image = decode_image_tiled(vq_model, code, tile_size=tile_size, overlap=4)
        
        # Extract any text (excluding image blocks)
        text_parts = re.split(pattern, output_text, flags=re.DOTALL)
        text_response = " ".join(t.strip() for t in text_parts if t.strip())
        
        return (pil_to_comfy(pil_image), text_response, "")


# =============================================================================
# Node: Emu35 Interleaved Generator (Story/HowTo)
# =============================================================================

class Emu35InterleavedGenerator:
    """
    Generate interleaved text and images for stories or tutorials.
    Requires the full Emu3.5 model (not Emu3.5-Image).
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "prompt": ("STRING", {"multiline": True, "default": "Tell me a story about a robot learning to paint"}),
                "task_type": (["story", "howto"],),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 10}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "full_text", "reasoning")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "Emu3.5"

    def generate(
        self,
        model,
        tokenizer,
        vq_model,
        prompt,
        task_type,
        cfg_scale,
        seed,
        max_images,
        reference_image=None,
    ):
        torch.manual_seed(seed)
        device = get_model_device(model)
        
        # Get task-specific sampling params
        preset = SAMPLING_PRESETS[task_type]
        
        # Build config
        cfg = Emu35Config(
            task_type=task_type,
            cfg_scale=cfg_scale,
            text_top_k=preset["text_top_k"],
            text_top_p=preset["text_top_p"],
            text_temperature=preset["text_temperature"],
            image_top_k=preset["image_top_k"],
            max_new_tokens=preset["max_new_tokens"],
        )
        
        # Build prompt
        if reference_image is not None:
            # Encode reference image
            image_token_string = encode_reference_images(
                [reference_image], vq_model, tokenizer
            )
            template = TASK_TEMPLATES[task_type]["with_image"]
            full_prompt = template.format(question=prompt)
            full_prompt = full_prompt.replace("<|IMAGE|>", image_token_string)
            
            unc_template = TASK_TEMPLATES[task_type]["unconditional"]
            unc_prompt = unc_template.replace("<|IMAGE|>", image_token_string)
        else:
            template = TASK_TEMPLATES[task_type]["without_image"]
            full_prompt = template.format(question=prompt)
            unc_prompt = TASK_TEMPLATES[task_type]["unconditional"]
        
        # Encode
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        print(f"[{task_type.upper()}] Generating interleaved content...")
        
        # Build logits processor
        logits_processor = LogitsProcessorList()
        logits_processor.append(
            build_logits_processor(
                cfg=cfg,
                unconditional_ids=unconditional_ids,
                model=model,
                tokenizer=tokenizer,
                force_same_image_size=True,
            )
        )
        
        # Generate
        generation_config = GenerationConfig(
            **cfg.sampling_params,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        
        # Custom stopping: stop after max_images (counting only GENERATED images, not input)
        class StopAfterNImages(StoppingCriteria):
            def __init__(self, max_images, eoi_token_id, input_length):
                self.max_images = max_images
                self.eoi_token_id = eoi_token_id
                self.input_length = input_length
                self.image_count = 0
                
            def __call__(self, input_ids, scores, **kwargs):
                # Count EOI tokens only in GENERATED part
                generated = input_ids[0, self.input_length:]
                self.image_count = (generated == self.eoi_token_id).sum().item()
                return self.image_count >= self.max_images
        
        stopping_criteria = StoppingCriteriaList([
            StopAfterNImages(max_images, EOI_TOKEN_ID, input_ids.shape[1])
        ])
        
        streamer = ProgressStreamer(cfg.sampling_params["max_new_tokens"])
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
        
        gen_time = time.time() - start_time
        print(f"[{task_type.upper()}] Generation took {gen_time:.1f}s")
        
        # Decode all images and text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        try:
            mm_output = multimodal_decode(output_text, tokenizer, vq_model)
            
            images = []
            full_text = ""
            reasoning = ""
            
            for item_type, item_data in mm_output:
                if item_type == "image" and item_data is not None:
                    images.append(pil_to_comfy(item_data))
                elif item_type == "text":
                    full_text += item_data + "\n"
                elif item_type == "global_cot":
                    reasoning += f"[Global] {item_data}\n"
                elif item_type == "image_cot":
                    reasoning += f"[Image] {item_data}\n"
            
            if images:
                return (images, full_text.strip(), reasoning)
            
        except Exception as e:
            print(f"[{task_type.upper()}] Decode error: {e}")
            import traceback
            traceback.print_exc()
        
        return ([torch.zeros(1, 512, 512, 3)], "", "")


# =============================================================================
# Node: Emu35 VQA (Visual Question Answering)
# =============================================================================

class Emu35VQA:
    """
    Visual Question Answering - describe or analyze images.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "answer"
    CATEGORY = "Emu3.5"

    def answer(self, model, tokenizer, vq_model, image, question, max_tokens):
        device = get_model_device(model)
        
        # Encode image
        image_token_string = encode_reference_images([image], vq_model, tokenizer)
        
        # Build prompt for understanding
        template = "<|extra_203|>You are a helpful assistant. USER: {question}<|IMAGE|> ASSISTANT: "
        full_prompt = template.format(question=question)
        full_prompt = full_prompt.replace("<|IMAGE|>", image_token_string)
        
        # Encode
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        print(f"[VQA] Processing question: {question[:50]}...")
        
        # Generate (no CFG needed for understanding)
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
            )
        
        # Decode only new tokens
        new_tokens = outputs[:, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        
        return (response,)


# =============================================================================
# Node: Emu35 Memory Manager
# =============================================================================

class Emu35MemoryManager:
    """
    Manage model memory - move between devices or clear VRAM.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "action": (["keep_in_vram", "move_to_ram", "clear_cache"],),
            },
            "optional": {
                "vq_model": ("EMU_VQ",),
            }
        }

    RETURN_TYPES = ("EMU_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "manage"
    CATEGORY = "Emu3.5"
    OUTPUT_NODE = True

    def manage(self, model, action, vq_model=None):
        if action == "move_to_ram":
            print("[Memory] Moving model to RAM...")
            model.to("cpu")
            if vq_model is not None:
                vq_model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            print("[Memory] Models moved to RAM")
            
        elif action == "clear_cache":
            print("[Memory] Clearing CUDA cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Report memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1024**3
                    print(f"[Memory] GPU {i}: {mem:.2f} GB allocated")
        
        return (model,)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "Emu35LoaderV2": Emu35LoaderV2,
    "Emu35T2ISamplerV2": Emu35T2ISamplerV2,
    "Emu35X2ISampler": Emu35X2ISampler,
    "Emu35InterleavedGenerator": Emu35InterleavedGenerator,
    "Emu35VQA": Emu35VQA,
    "Emu35MemoryManager": Emu35MemoryManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35LoaderV2": "Emu 3.5 Loader V2",
    "Emu35T2ISamplerV2": "Emu 3.5 T2I Sampler V2",
    "Emu35X2ISampler": "Emu 3.5 X2I (Image Edit)",
    "Emu35InterleavedGenerator": "Emu 3.5 Interleaved (Story/HowTo)",
    "Emu35VQA": "Emu 3.5 VQA",
    "Emu35MemoryManager": "Emu 3.5 Memory Manager",
}
