import argparse
import math
import os
import shutil
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# Add the Emu3.5 repo to sys.path if it exists locally
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
if os.path.exists(emu_repo_path):
    sys.path.append(emu_repo_path)
    print(f"Added {emu_repo_path} to sys.path")

# Register Emu3 architecture with Auto classes (if not already present)
EmuModelClass = None
try:
    from src.emu3p5.configuration_emu3 import Emu3Config
    from src.emu3p5.modeling_emu3 import Emu3ForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    print("Registering Emu3 architecture...")
    try:
        AutoConfig.register("Emu3", Emu3Config)
    except ValueError:
        pass  # Already registered in this interpreter
    try:
        AutoModelForCausalLM.register(Emu3Config, Emu3ForCausalLM)
    except ValueError:
        pass
    EmuModelClass = Emu3ForCausalLM
    print("Successfully ensured Emu3 is registered with AutoModel.")
except ImportError as e:
    print(f"Warning: Could not import Emu3 classes for registration: {e}")
    print("Will rely on trust_remote_code=True and hope the model folder contains the code.")

def quantize_model(model_path, output_path, quant_type="nf4", skip_modules=None):
    print(f"Loading model from {model_path}...")
    
    if skip_modules is None:
        skip_modules = []
        
    print(f"Skipping quantization for modules: {skip_modules}")
    
    # Match Hunyuan script exactly - simple config, no device_map complexities
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        modules_to_not_convert=skip_modules,
        llm_int8_skip_modules=skip_modules
    )

    model = None
    
    # Critical: Use device_map="cuda:0" like Hunyuan, NOT "auto"
    # This forces GPU-only placement and lets BitsAndBytes handle offload internally
    # "auto" causes Accelerate to pre-analyze placement and load shards into CPU RAM
    base_load_kwargs = dict(
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Force GPU placement - BnB handles the rest
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print("Loading with AutoModelForCausalLM (device_map='cuda:0')...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **base_load_kwargs)
        print("Successfully loaded model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have the Emu3.5 code available or the model repo is correct.")
        return

    print("Model loaded. Verifying quantization...")
    if hasattr(model, "hf_device_map"):
        print(f"Model device map: {model.hf_device_map}")

    print(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)

    # Ensure tokenization_emu3.py is present so AutoTokenizer works when loading the quantized model
    try:
        repo_tokenizer_file = os.path.join(emu_repo_path, "src", "tokenizer_emu3_ibq", "tokenization_emu3.py")
        model_tokenizer_file = os.path.join(model_path, "tokenization_emu3.py")
        dest_tokenizer_file = os.path.join(output_path, "tokenization_emu3.py")

        source_file = None
        if repo_tokenizer_file and os.path.exists(repo_tokenizer_file):
            source_file = repo_tokenizer_file
        elif os.path.exists(model_tokenizer_file):
            source_file = model_tokenizer_file

        if source_file is not None:
            shutil.copy(source_file, dest_tokenizer_file)
            print("Copied tokenization_emu3.py into quantized folder.")
        else:
            print("Warning: Could not find tokenization_emu3.py to copy alongside quantized model.")
    except Exception as e:
        print(f"Warning: Failed to copy tokenization_emu3.py: {e}")
    
    # Also save tokenizer if present
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Could not save tokenizer: {e}")

    print("Quantization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Emu 3.5 model to 4-bit (NF4)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the quantized model")
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["nf4", "fp4"], help="Quantization type")
    parser.add_argument("--skip_modules", nargs="+", default=[], help="List of module names to skip quantization (e.g. lm_head)")
    parser.add_argument("--cpu_memory_gb", type=int, default=256, help="CPU memory budget for streaming load (default: 256 GiB)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    quantize_model(args.model_path, args.output_path, args.quant_type, args.skip_modules, args.cpu_memory_gb)
