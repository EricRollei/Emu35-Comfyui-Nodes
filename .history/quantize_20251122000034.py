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

def quantize_model(model_path, output_path, quant_type="nf4"):
    print(f"Loading model from {model_path}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Try loading with trust_remote_code=True to handle custom models like Emu3
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have the Emu3.5 code available or the model repo is correct.")
        return

    print(f"Model loaded. Saving quantized model to {output_path}...")
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    quantize_model(args.model_path, args.output_path, args.quant_type)
