import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# Add the Emu3.5 repo to sys.path if it exists locally
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
if os.path.exists(emu_repo_path):
    sys.path.append(emu_repo_path)
    print(f"Added {emu_repo_path} to sys.path")

# Try to import Emu3 class directly
try:
    from src.emu3p5.modeling_emu3 import Emu3ForCausalLM
    print("Successfully imported Emu3ForCausalLM from local repo.")
    ModelClass = Emu3ForCausalLM
except ImportError as e:
    print(f"Could not import Emu3ForCausalLM: {e}")
    print("Falling back to AutoModelForCausalLM")
    ModelClass = AutoModelForCausalLM

def quantize_model(model_path, output_path, quant_type="nf4", skip_modules=None):
    print(f"Loading model from {model_path}...")
    
    if skip_modules is None:
        skip_modules = []
        
    print(f"Skipping quantization for modules: {skip_modules}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=skip_modules
    )

    # Try loading with trust_remote_code=True to handle custom models like Emu3
    try:
        # Enable low_cpu_mem_usage to avoid loading entire model into RAM
        model = ModelClass.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_state_dict=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have the Emu3.5 code available or the model repo is correct.")
        return

    print("Model loaded. Verifying quantization...")
    # Check a few layers
    for name, module in model.named_modules():
        if name in ["model.layers.0.self_attn.q_proj", "lm_head"]:
            print(f"Layer {name}: {type(module)}")

    print(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_path)
    
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    quantize_model(args.model_path, args.output_path, args.quant_type, args.skip_modules)
