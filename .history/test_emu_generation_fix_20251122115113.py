import sys
import os
import torch
from unittest.mock import MagicMock

# Add repo to path
current_dir = os.path.dirname(os.path.abspath(__file__))
emu_repo_path = os.path.join(current_dir, "Emu3_5_repo")
sys.path.append(emu_repo_path)

# Mock transformers.cache_utils.Cache and DynamicCache
# We need to do this BEFORE importing modeling_emu3 because it imports them
import transformers.cache_utils

# Create a Mock DynamicCache that mimics the user's environment
# Missing: get_max_length, get_usable_length, seen_tokens
# Present: get_seq_length, max_cache_len
class MockDynamicCache(transformers.cache_utils.Cache):
    def __init__(self):
        # self.max_cache_len = 1024 # Cannot set property
        pass
    
    @property
    def max_cache_len(self):
        return 1024

    def get_seq_length(self, layer_idx=0):
        return 10

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

# Monkey patch transformers
transformers.cache_utils.DynamicCache = MockDynamicCache

try:
    from src.emu3p5.modeling_emu3 import Emu3ForCausalLM, Emu3Config, Emu3Attention
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_prepare_inputs_for_generation():
    print("Testing prepare_inputs_for_generation...")
    config = Emu3Config()
    model = Emu3ForCausalLM(config)
    
    input_ids = torch.zeros((1, 20), dtype=torch.long)
    past_key_values = MockDynamicCache()
    
    # This should NOT crash
    try:
        model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        print("SUCCESS: prepare_inputs_for_generation passed.")
    except AttributeError as e:
        print(f"FAILURE: prepare_inputs_for_generation failed with {e}")
    except Exception as e:
        print(f"FAILURE: prepare_inputs_for_generation failed with unexpected error {e}")

def test_attention_forward():
    print("Testing Emu3Attention forward...")
    config = Emu3Config()
    attn = Emu3Attention(config, layer_idx=0)
    
    hidden_states = torch.randn(1, 10, config.hidden_size)
    past_key_value = MockDynamicCache()
    
    # This should NOT crash
    try:
        attn(hidden_states, past_key_value=past_key_value)
        print("SUCCESS: Emu3Attention forward passed.")
    except AttributeError as e:
        print(f"FAILURE: Emu3Attention forward failed with {e}")
    except Exception as e:
        print(f"FAILURE: Emu3Attention forward failed with unexpected error {e}")

if __name__ == "__main__":
    test_prepare_inputs_for_generation()
    test_attention_forward()
