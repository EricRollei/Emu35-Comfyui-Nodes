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
# CRITICAL: Patch DynamicCache BEFORE importing Emu3.5 modules
# Transformers 4.50+ removed 'seen_tokens' attribute from DynamicCache
# The Emu3 model code expects this attribute to exist
# =============================================================================
try:
    from transformers.cache_utils import DynamicCache
    
    # Patch 1: Add get_usable_length if missing (removed in newer transformers)
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, input_seq_length, layer_idx=None):
            # Call get_seq_length without layer_idx if it's None
            if layer_idx is not None:
                return self.get_seq_length(layer_idx)
            else:
                return self.get_seq_length()
        DynamicCache.get_usable_length = _get_usable_length
        print("[Emu35] Patched DynamicCache.get_usable_length")
    
    # Patch 2: Add seen_tokens property if missing (removed in newer transformers)
    if not hasattr(DynamicCache, "seen_tokens"):
        @property
        def _seen_tokens(self):
            return self.get_seq_length()
        DynamicCache.seen_tokens = _seen_tokens
        print("[Emu35] Patched DynamicCache.seen_tokens")
        
except ImportError as e:
    print(f"[Emu35] Warning: Could not patch DynamicCache: {e}")

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
                    # Skip hidden folders (start with .) and offload cache
                    if not item.startswith('.') and 'offload_cache' not in item.lower():
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

        # Determine device and device_map
        if device == "auto":
            # Use accelerate's auto placement which handles multi-GPU splitting
            llm_device_map = "auto"
            # For operations requiring a specific device (like VQ or initial tensor creation), use cuda:0
            execution_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            # User forced a specific device
            llm_device_map = {"": device}
            execution_device = device
        
        if vq_device == "same":
            vq_device = execution_device

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
        print(f"  Device Selection: {device} -> Map: {llm_device_map}")
        print(f"  Execution Device: {execution_device}, VQ Device: {vq_device}")
        print(f"  Precision: {precision} -> {load_mode}")

        # Load Config - use LOCAL Emu3Config to avoid HuggingFace cache
        try:
            config = Emu3Config.from_pretrained(model_path)
            from transformers import AutoConfig
            AutoConfig.register("Emu3", Emu3Config)
        except Exception as e:
            print(f"Warning: Could not load Emu3Config ({e}), trying AutoConfig...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Use eager attention (sdpa has issues on Blackwell)
        attn_impl = "eager"

        # Check if this is a sharded model (multiple safetensors files)
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        is_sharded = len(safetensors_files) > 1
        print(f"  Sharded model: {is_sharded} ({len(safetensors_files)} files)")
        
        # Load Model
        if load_mode == "pre_quantized":
            model = Emu3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=llm_device_map,
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
                device_map=llm_device_map,
                attn_implementation=attn_impl
            )
        else:
            # Full-precision load (BF16/FP32)
            # Hybrid Strategy: Use Manual Patching for 'base' model, Simple Load for others (Image)
            
            is_base_model = "base" in model_name.lower()
            
            if not is_base_model:
                # --- SIMPLE STRATEGY (V1 Style) for Image Model ---
                print(f"  Loading model from: {model_path}")
                print(f"  Using Local Emu3ForCausalLM (no HuggingFace cache)")
                
                # Use LOCAL Emu3ForCausalLM to avoid HuggingFace cache issues
                model = Emu3ForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=dtype,
                    device_map=llm_device_map,
                    attn_implementation=attn_impl
                )
                
                # Note: DynamicCache patches are now applied at module load time (top of file)
                # We still patch prepare_inputs_for_generation as a safety measure
                import types
                def patched_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
                    if past_key_values:
                        if hasattr(past_key_values, "get_seq_length"):
                            past_length = past_key_values.get_seq_length()
                        elif hasattr(past_key_values, "seen_tokens"):
                            past_length = past_key_values.seen_tokens
                        else:
                            past_length = past_key_values[0][0].shape[2]
                    else:
                        past_length = 0

                    if past_length > 0:
                        input_ids = input_ids[:, past_length:]

                    position_ids = kwargs.get("position_ids", None)
                    if attention_mask is not None and position_ids is None:
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 1)
                        if past_length > 0:
                            position_ids = position_ids[:, past_length:]

                    model_inputs = {
                        "input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids,
                        "attention_mask": attention_mask,
                    }
                    return model_inputs

                # Apply patch to INSTANCE and CLASS to be safe
                model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, model)
                if hasattr(model, "__class__"):
                     model.__class__.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
                
                print("  ✓ Applied 'seen_tokens' compatibility patch (Image Model)")
                
            else:
                # --- MANUAL PATCHING STRATEGY for Base Model ---
                print(f"  Loading model from: {model_path}")
                print(f"  Using Manual Patching Strategy (init_empty_weights + manual fix) for Base model")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    from accelerate import load_checkpoint_in_model, init_empty_weights
                    from safetensors.torch import load_file
                except ImportError:
                    raise ImportError("accelerate and safetensors are required. Please install them.")

                # 1. Get Config and Class
                config = Emu3Config.from_pretrained(model_path)
                # Force eager attention
                config._attn_implementation = attn_impl
                
                # 2. Init Empty Model
                print("  Initializing empty model structure...")
                with init_empty_weights():
                    # Use LOCAL Emu3ForCausalLM to avoid HuggingFace cache
                    model = Emu3ForCausalLM(config)
                
                # 3. Load Checkpoint via Accelerate
                if is_sharded:
                    checkpoint_path = model_path
                else:
                    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") or f.endswith(".bin")]
                    checkpoint_path = os.path.join(model_path, files[0])

                print(f"  Loading weights via accelerate from {checkpoint_path}...")
                load_checkpoint_in_model(
                    model,
                    checkpoint_path,
                    device_map=llm_device_map,
                    dtype=dtype
                )
                
                # 4. Manual Patching of Missing Weights
                meta_params = [name for name, param in model.named_parameters() if param.device.type == 'meta']
                
                if meta_params:
                    print(f"  WARNING: {len(meta_params)} parameters still on meta device. Attempting manual patch...")
                    
                    from safetensors import safe_open
                    
                    # Build a fresh index for the missing parameters only
                    print("  Scanning safetensors files to locate missing weights...")
                    found_weights = {} # name -> filename
                    
                    safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
                    
                    for fname in safetensors_files:
                        fpath = os.path.join(model_path, fname)
                        try:
                            with safe_open(fpath, framework="pt") as f:
                                file_keys = set(f.keys())
                                
                            for name in meta_params:
                                if name in found_weights:
                                    continue
                                    
                                # Check exact match
                                if name in file_keys:
                                    found_weights[name] = fname
                                    continue
                                    
                                # Check with/without 'model.' prefix
                                if name.startswith("model."):
                                    short_name = name[6:]
                                    if short_name in file_keys:
                                        found_weights[name] = fname
                                        continue
                                else:
                                    long_name = f"model.{name}"
                                    if long_name in file_keys:
                                        found_weights[name] = fname
                                        continue
                                        
                        except Exception as e:
                            print(f"    Error scanning {fname}: {e}")

                    # Group by file for efficient loading
                    params_by_file = {}
                    for name, fname in found_weights.items():
                        if fname not in params_by_file:
                            params_by_file[fname] = []
                        params_by_file[fname].append(name)
                    
                    # Report still missing
                    still_missing = set(meta_params) - set(found_weights.keys())
                    if still_missing:
                        print(f"  ERROR: Could not find {len(still_missing)} weights in any file: {list(still_missing)[:5]}...")

                    # Load and patch
                    for fname, names in params_by_file.items():
                        fpath = os.path.join(model_path, fname)
                        print(f"    Patching {len(names)} weights from {fname}...")
                        
                        try:
                            # Load shard
                            state_dict = load_file(fpath)
                            keys_in_file = set(state_dict.keys())
                            
                            for name in names:
                                # Determine the key used in the file
                                target_key = name
                                if target_key not in keys_in_file:
                                    if name.startswith("model.") and name[6:] in keys_in_file:
                                        target_key = name[6:]
                                    elif f"model.{name}" in keys_in_file:
                                        target_key = f"model.{name}"
                                
                                if target_key in state_dict:
                                    # Move to device and cast
                                    tensor = state_dict[target_key].to(device=execution_device, dtype=dtype)
                                    
                                    # Find the parameter object
                                    if "." in name:
                                        module_path, param_name = name.rsplit(".", 1)
                                        submod = model.get_submodule(module_path)
                                    else:
                                        submod = model
                                        param_name = name
                                    
                                    # Get original param to check requires_grad
                                    param = getattr(submod, param_name)
                                    
                                    # Create new parameter on device
                                    new_param = torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
                                    setattr(submod, param_name, new_param)
                                    
                                else:
                                    print(f"      Error: {name} (target: {target_key}) not found in {fname}")
                            
                            del state_dict
                        except Exception as e:
                            print(f"    Error patching from {fname}: {e}")
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Tie weights
                model.tie_weights()
                
                # Note: DynamicCache patches are now applied at module load time (top of file)
                # Patch prepare_inputs_for_generation for compatibility
                import types
                def patched_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
                    # This is a simplified version of what Emu3 likely expects, adapted for newer transformers
                    if past_key_values:
                        if hasattr(past_key_values, "get_seq_length"):
                            past_length = past_key_values.get_seq_length()
                        elif hasattr(past_key_values, "seen_tokens"):
                            past_length = past_key_values.seen_tokens
                        else:
                            past_length = past_key_values[0][0].shape[2]
                    else:
                        past_length = 0

                    # Standard logic from here (simplified for Emu3 context)
                    if past_length > 0:
                        input_ids = input_ids[:, past_length:]

                    position_ids = kwargs.get("position_ids", None)
                    if attention_mask is not None and position_ids is None:
                        # create position_ids on the fly for batch generation
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 1)
                        if past_length > 0:
                            position_ids = position_ids[:, past_length:]

                    model_inputs = {
                        "input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids,
                        "attention_mask": attention_mask,
                    }
                    return model_inputs

                # Apply the patch to the model instance AND CLASS
                model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, model)
                if hasattr(model, "__class__"):
                     model.__class__.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
                
                print("  ✓ Applied 'seen_tokens' compatibility patch")

            # Inject GenerationMixin
            from transformers.generation.utils import GenerationMixin
            if not isinstance(model, GenerationMixin):
                print("  Injecting GenerationMixin into model...")
                model.__class__.__bases__ = (GenerationMixin,) + model.__class__.__bases__
            
            # Config
            if model.generation_config is None:
                try:
                    model.generation_config = GenerationConfig.from_pretrained(model_path)
                except:
                    model.generation_config = GenerationConfig(
                        bos_token_id=BOS_TOKEN_ID,
                        eos_token_id=EOS_TOKEN_ID,
                        pad_token_id=PAD_TOKEN_ID,
                    )
            
            print("  ✓ Model loaded successfully")

            # Verify model loaded correctly
            sample_param = next(model.parameters())
            print(f"  ✓ Model loaded to: {sample_param.device}, dtype: {sample_param.dtype}")
        
        model.eval()
        
        # Report which device(s) the model ended up on
        if hasattr(model, 'hf_device_map'):
            devices_used = set(model.hf_device_map.values())
            print(f"  Model distributed across: {devices_used}")
        
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
    
    ⚠️ IMPORTANT: This node requires the BASE Emu3.5 model, NOT Emu3.5-Image!
    
    - Emu3.5-Image: Trained for T2I and X2I tasks only
    - Emu3.5 (base): Trained for interleaved text+image generation (story, howto)
    
    Download base model:
        huggingface-cli download BAAI/Emu3.5 --local-dir models/emu35/Emu3.5
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
        
        # Check model type - warn if using Emu3.5-Image for story/howto
        # Emu3.5-Image is for T2I/X2I only; story/howto needs base Emu3.5
        model_config = getattr(model, 'config', None)
        if model_config:
            model_name = getattr(model_config, '_name_or_path', '')
            if 'image' in model_name.lower() or 'Image' in model_name:
                print(f"\n[{task_type.upper()}] ⚠️  WARNING: You appear to be using Emu3.5-Image model")
                print(f"[{task_type.upper()}] Story/HowTo generation works best with the BASE Emu3.5 model")
                print(f"[{task_type.upper()}] The Emu3.5-Image model is optimized for T2I/X2I tasks only")
                print(f"[{task_type.upper()}] Download base model: huggingface-cli download BAAI/Emu3.5\n")
        
        # Get task-specific sampling params
        preset = SAMPLING_PRESETS[task_type]
        
        print(f"[{task_type.upper()}] max_new_tokens: {preset['max_new_tokens']}, max_images: {max_images}")
        
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
    
    Typical use cases:
    - Image captioning: "Describe this image in detail."
    - Object identification: "What objects are in this image?"
    - Scene understanding: "What is happening in this scene?"
    - OCR/Text reading: "What text appears in this image?"
    - Counting: "How many people are in this image?"
    - Spatial reasoning: "What is to the left of the car?"
    - Style analysis: "What artistic style is this image?"
    - Comparison: (use multiple images) "What's different between these images?"
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EMU_MODEL",),
                "tokenizer": ("EMU_TOKENIZER",),
                "vq_model": ("EMU_VQ",),
                "image": ("IMAGE",),
                "task_type": (["caption", "describe", "analyze", "ocr", "question", "compare", "custom"],),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "question": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter your question (required for 'question' and 'custom' modes)"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                "image_resolution": (["256x256", "384x384", "512x512"],),  # Lower res for VQA = less VRAM
                "image2": ("IMAGE",),  # Optional second image for comparison
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "answer"
    CATEGORY = "Emu3.5"

    def answer(self, model, tokenizer, vq_model, image, task_type, max_tokens, 
               question="", temperature=0.3, image_resolution="384x384", image2=None):
        device = get_model_device(model)
        
        # Parse image resolution to area
        res_map = {
            "256x256": 256 * 256,    # 65536 - very fast, lower quality
            "384x384": 384 * 384,    # 147456 - good balance
            "512x512": 512 * 512,    # 262144 - higher quality
        }
        image_area = res_map.get(image_resolution, 384 * 384)
        
        # Clear cache before encoding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Encode image(s) with reduced resolution for VQA
        num_images = 1 if image2 is None else 2
        
        print(f"[VQA] Encoding {num_images} image(s) at {image_resolution}...")
        
        # For multi-image, encode separately and label them
        if num_images == 2:
            image1_tokens = encode_reference_images([image], vq_model, tokenizer, image_area=image_area)
            image2_tokens = encode_reference_images([image2], vq_model, tokenizer, image_area=image_area)
            # Create labeled image tokens
            labeled_image_string = f"[First Image:] {image1_tokens} [Second Image:] {image2_tokens}"
            print(f"[VQA] Multi-image mode: Using labeled image tokens")
        else:
            labeled_image_string = encode_reference_images([image], vq_model, tokenizer, image_area=image_area)
        
        # Clear cache after encoding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Build task-specific prompts
        if num_images == 1:
            task_prompts = {
                "caption": "Provide a one-sentence caption for this image.",
                "describe": "Describe this image in detail, including the main subjects, background, colors, and overall composition.",
                "analyze": "Analyze this image thoroughly. Describe what you see, the context, mood, and any notable details.",
                "ocr": "Read and transcribe all text visible in this image.",
                "question": question,  # Use user's question directly
                "compare": "Describe this image in detail.",  # Fallback for single image
                "custom": question,
            }
        else:
            # Multi-image prompts - be very explicit about which image is which
            task_prompts = {
                "caption": "You are looking at two images. The first image is labeled [First Image] and the second is labeled [Second Image]. Provide a brief caption describing both.",
                "describe": "You are looking at two images. The first image is labeled [First Image] and the second is labeled [Second Image]. Describe what you see in each image separately.",
                "analyze": "You are looking at two images. The first image is labeled [First Image] and the second is labeled [Second Image]. Analyze each image and explain what they show.",
                "ocr": "You are looking at two images. Read and transcribe all text visible in both images.",
                "question": question,
                "compare": "You are looking at two images. The first image is labeled [First Image] and the second is labeled [Second Image]. Compare these two images: What is in the first image? What is in the second image? What are the differences?",
                "custom": question,
            }
        
        actual_question = task_prompts.get(task_type, question)
        
        # For 'question' and 'custom' modes, require user input
        if task_type in ["question", "custom"]:
            if not question.strip():
                return ("Error: Please enter a question or prompt for 'question' or 'custom' mode.",)
            actual_question = question
        else:
            # For preset tasks, append user question if they provided one
            if question.strip():
                actual_question = f"{actual_question} Also: {question}"
        
        # Official Emu3.5 template for understanding tasks
        # Image comes FIRST, then the question - this is the proper VQA format
        # <|extra_100|> (BSS token) signals the model to start generating
        template = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> {question} ASSISTANT: <|extra_100|>"
        full_prompt = template.format(question=actual_question)
        full_prompt = full_prompt.replace("<|IMAGE|>", labeled_image_string)
        
        # Encode
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        print(f"[VQA] Task: {task_type}, Images: {num_images}")
        print(f"[VQA] Question: {actual_question[:80]}...")
        print(f"[VQA] Input tokens: {input_ids.shape[1]}")
        
        # Generate - understanding tasks use lower temperature for accuracy
        # Include repetition penalty and no_repeat_ngram to prevent loops
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            top_k=200 if temperature > 0 else 1,  # Lower top_k for understanding
            top_p=0.8,
            temperature=max(temperature, 0.01),  # Avoid division by zero
            repetition_penalty=1.2,  # Penalize repeated tokens
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition (like "and, and, and")
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
        
        # Clean up response - remove any trailing special tokens that weren't caught
        response = response.strip()
        
        # Remove common artifacts
        for artifact in ["<|extra_101|>", "<|extra_100|>", "<|endoftext|>"]:
            response = response.replace(artifact, "").strip()
        
        print(f"[VQA] Response length: {len(response)} chars")
        
        # Clean up after generation
        del input_ids, outputs, new_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (response,)


# =============================================================================
# Node: Emu35 Memory Manager (Enhanced)
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
# Node: Emu35 VRAM Cleanup (Comprehensive)
# =============================================================================

class Emu35VRAMCleanup:
    """
    Comprehensive VRAM and RAM cleanup node.
    
    Can be wired anywhere in workflow:
    - Before Emu35 nodes: Clean slate for generation
    - After Emu35 nodes: Free up VRAM for other tasks
    - Standalone: Emergency cleanup after OOM errors
    
    Features:
    - Reports VRAM/RAM before and after
    - Multiple cleanup levels (light, standard, aggressive)
    - Can unload model weights entirely
    - Cleans up orphaned tensors from OOM situations
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cleanup_level": ([
                    "light (cache only)",
                    "standard (cache + gc)", 
                    "aggressive (deep clean)",
                    "nuclear (unload all models)"
                ],),
                "trigger": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Accept any input type for flexible wiring
                "any_input": ("*",),
                "model": ("EMU_MODEL",),
                "vq_model": ("EMU_VQ",),
                "unload_models": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*", "STRING", "STRING")
    RETURN_NAMES = ("passthrough", "status_report", "memory_diff")
    FUNCTION = "cleanup"
    CATEGORY = "Emu3.5"
    OUTPUT_NODE = True

    def _get_memory_stats(self) -> dict:
        """Collect current memory statistics."""
        import psutil
        
        stats = {
            "ram_used_gb": psutil.Process().memory_info().rss / 1024**3,
            "ram_available_gb": psutil.virtual_memory().available / 1024**3,
            "ram_total_gb": psutil.virtual_memory().total / 1024**3,
            "gpus": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    free = total - reserved
                    stats["gpus"].append({
                        "id": i,
                        "name": torch.cuda.get_device_properties(i).name,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "free_gb": free,
                        "total_gb": total,
                    })
                except Exception as e:
                    stats["gpus"].append({"id": i, "error": str(e)})
        
        return stats

    def _format_stats(self, stats: dict, label: str) -> str:
        """Format memory stats as readable string."""
        lines = [f"=== {label} ==="]
        lines.append(f"RAM: {stats['ram_used_gb']:.2f}GB used / {stats['ram_total_gb']:.1f}GB total")
        
        for gpu in stats["gpus"]:
            if "error" in gpu:
                lines.append(f"GPU {gpu['id']}: Error - {gpu['error']}")
            else:
                lines.append(
                    f"GPU {gpu['id']} ({gpu['name']}): "
                    f"{gpu['allocated_gb']:.2f}GB allocated, "
                    f"{gpu['reserved_gb']:.2f}GB reserved, "
                    f"{gpu['free_gb']:.2f}GB free"
                )
        
        return "\n".join(lines)

    def _light_cleanup(self):
        """Light cleanup - just clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _standard_cleanup(self):
        """Standard cleanup - cache + garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _aggressive_cleanup(self):
        """Aggressive cleanup - deep clean for orphaned tensors."""
        import ctypes
        
        # Multiple rounds of GC
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            # Synchronize all devices
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)
                except:
                    pass
            
            # Clear caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset peak memory stats
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                except:
                    pass
        
        # Try to release Python memory back to OS
        try:
            if hasattr(ctypes, 'pythonapi'):
                ctypes.pythonapi.PyMem_RawFree
            # On Windows, try to compact heap
            import sys
            if sys.platform == 'win32':
                try:
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except:
                    pass
        except:
            pass

    def _nuclear_cleanup(self, model=None, vq_model=None):
        """Nuclear option - unload models entirely."""
        unloaded = []
        
        # Move models to CPU first, then delete references
        if model is not None:
            try:
                model.to("cpu")
                unloaded.append("main model → CPU")
            except Exception as e:
                unloaded.append(f"main model: error {e}")
        
        if vq_model is not None:
            try:
                vq_model.to("cpu")
                unloaded.append("VQ model → CPU")
            except Exception as e:
                unloaded.append(f"VQ model: error {e}")
        
        # Aggressive cleanup after moving
        self._aggressive_cleanup()
        
        return unloaded

    def cleanup(
        self, 
        cleanup_level: str,
        trigger: bool,
        any_input=None,
        model=None, 
        vq_model=None,
        unload_models: bool = False,
    ):
        # Get before stats
        before_stats = self._get_memory_stats()
        before_report = self._format_stats(before_stats, "BEFORE CLEANUP")
        
        print(f"\n[VRAM Cleanup] Level: {cleanup_level}")
        print(before_report)
        
        # Only run if triggered
        if not trigger:
            return (any_input, "Cleanup skipped (trigger=False)", "No change")
        
        unloaded_info = []
        
        # Perform cleanup based on level
        if "light" in cleanup_level:
            self._light_cleanup()
        elif "standard" in cleanup_level:
            self._standard_cleanup()
        elif "aggressive" in cleanup_level:
            self._aggressive_cleanup()
        elif "nuclear" in cleanup_level:
            unloaded_info = self._nuclear_cleanup(model, vq_model)
        
        # Additional model unload if requested
        if unload_models and "nuclear" not in cleanup_level:
            unloaded_info = self._nuclear_cleanup(model, vq_model)
        
        # Get after stats
        after_stats = self._get_memory_stats()
        after_report = self._format_stats(after_stats, "AFTER CLEANUP")
        
        # Calculate diff
        diff_lines = ["=== MEMORY FREED ==="]
        ram_freed = before_stats["ram_used_gb"] - after_stats["ram_used_gb"]
        diff_lines.append(f"RAM: {ram_freed:+.2f}GB")
        
        for i, (before_gpu, after_gpu) in enumerate(zip(before_stats["gpus"], after_stats["gpus"])):
            if "error" not in before_gpu and "error" not in after_gpu:
                alloc_freed = before_gpu["allocated_gb"] - after_gpu["allocated_gb"]
                reserved_freed = before_gpu["reserved_gb"] - after_gpu["reserved_gb"]
                diff_lines.append(
                    f"GPU {i}: {alloc_freed:+.2f}GB allocated, {reserved_freed:+.2f}GB reserved"
                )
        
        if unloaded_info:
            diff_lines.append("Models unloaded: " + ", ".join(unloaded_info))
        
        diff_report = "\n".join(diff_lines)
        
        # Full status report
        status_report = f"{before_report}\n\n{after_report}\n\n{diff_report}"
        
        print(after_report)
        print(diff_report)
        print("[VRAM Cleanup] Complete\n")
        
        return (any_input, status_report, diff_report)


# =============================================================================
# Node: Emu35 Emergency Reset
# =============================================================================

class Emu35EmergencyReset:
    """
    Emergency reset node for recovering from OOM or stuck states.
    
    This node attempts to:
    1. Find and clean up orphaned tensors
    2. Reset CUDA contexts
    3. Force garbage collection
    4. Clear all caches
    
    Use after OOM errors or when VRAM appears stuck.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "confirm_reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "target_gpu": ("INT", {"default": 0, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("reset_log",)
    FUNCTION = "emergency_reset"
    CATEGORY = "Emu3.5"
    OUTPUT_NODE = True

    def emergency_reset(self, confirm_reset: bool, target_gpu: int = 0):
        if not confirm_reset:
            return ("Reset cancelled - set confirm_reset=True to proceed",)
        
        log_lines = ["=== EMERGENCY RESET INITIATED ==="]
        log_lines.append(f"Target GPU: {target_gpu}")
        
        try:
            import psutil
            
            # Step 1: Get initial state
            if torch.cuda.is_available():
                initial_alloc = torch.cuda.memory_allocated(target_gpu) / 1024**3
                initial_reserved = torch.cuda.memory_reserved(target_gpu) / 1024**3
                log_lines.append(f"Initial: {initial_alloc:.2f}GB alloc, {initial_reserved:.2f}GB reserved")
            
            # Step 2: Synchronize GPU to finish pending ops
            log_lines.append("Step 1: Synchronizing CUDA...")
            if torch.cuda.is_available():
                torch.cuda.synchronize(target_gpu)
            log_lines.append("  ✓ CUDA synchronized")
            
            # Step 3: Multiple garbage collection passes
            log_lines.append("Step 2: Aggressive garbage collection...")
            collected_total = 0
            for i in range(5):
                collected = gc.collect()
                collected_total += collected
            log_lines.append(f"  ✓ Collected {collected_total} objects")
            
            # Step 4: Clear all CUDA caches
            log_lines.append("Step 3: Clearing CUDA caches...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats(target_gpu)
                torch.cuda.reset_accumulated_memory_stats(target_gpu)
            log_lines.append("  ✓ CUDA caches cleared")
            
            # Step 5: Try to find orphaned tensors in globals
            log_lines.append("Step 4: Scanning for orphaned tensors...")
            orphan_count = 0
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        # Check if tensor is unreachable (no strong refs except gc)
                        if len(gc.get_referrers(obj)) <= 1:
                            orphan_count += 1
                except:
                    pass
            log_lines.append(f"  Found {orphan_count} potentially orphaned CUDA tensors")
            
            # Step 6: Final GC pass
            log_lines.append("Step 5: Final cleanup...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 7: Report final state
            if torch.cuda.is_available():
                final_alloc = torch.cuda.memory_allocated(target_gpu) / 1024**3
                final_reserved = torch.cuda.memory_reserved(target_gpu) / 1024**3
                freed_alloc = initial_alloc - final_alloc
                freed_reserved = initial_reserved - final_reserved
                log_lines.append(f"\n=== RESULTS ===")
                log_lines.append(f"Final: {final_alloc:.2f}GB alloc, {final_reserved:.2f}GB reserved")
                log_lines.append(f"Freed: {freed_alloc:.2f}GB alloc, {freed_reserved:.2f}GB reserved")
            
            log_lines.append("\n✓ Emergency reset complete")
            
        except Exception as e:
            log_lines.append(f"\n✗ Error during reset: {type(e).__name__}: {e}")
            import traceback
            log_lines.append(traceback.format_exc())
        
        log_text = "\n".join(log_lines)
        print(log_text)
        return (log_text,)


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
    "Emu35VRAMCleanup": Emu35VRAMCleanup,
    "Emu35EmergencyReset": Emu35EmergencyReset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu35LoaderV2": "Emu 3.5 Loader V2",
    "Emu35T2ISamplerV2": "Emu 3.5 T2I Sampler V2",
    "Emu35X2ISampler": "Emu 3.5 X2I (Image Edit)",
    "Emu35InterleavedGenerator": "Emu 3.5 Interleaved (Story/HowTo)",
    "Emu35VQA": "Emu 3.5 VQA",
    "Emu35MemoryManager": "Emu 3.5 Memory Manager",
    "Emu35VRAMCleanup": "Emu 3.5 VRAM Cleanup",
    "Emu35EmergencyReset": "Emu 3.5 Emergency Reset",
}
