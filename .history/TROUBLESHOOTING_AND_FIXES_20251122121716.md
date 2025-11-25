# Emu 3.5 ComfyUI Node Troubleshooting & Fixes

## Overview
This document records the compatibility issues and bugs encountered when porting Emu 3.5 to ComfyUI, specifically when running in an environment with newer `transformers` versions (v4.56.0) compared to the repository's requirement (v4.48.2).

## 1. Upstream Bug in SDPA Attention
**File:** `src/emu3p5/modeling_emu3.py`
**Class:** `Emu3SdpaAttention`

**Issue:**
The official repository contains a logic error in the Scaled Dot Product Attention (SDPA) implementation. While the "Eager" and "Flash Attention" implementations correctly reshape the attention output to `-1` (inferring the size), the SDPA implementation hardcoded the reshape to `self.hidden_size` (5120).
For this model configuration, the internal dimensions (`num_heads * head_dim`) result in a size of 8192, causing a shape mismatch error:
`RuntimeError: shape '[1, 301, 5120]' is invalid for input of size 2465792`

**Fix:**
Modified `modeling_emu3.py` to use dynamic reshaping:
```python
# Old
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

# New
attn_output = attn_output.reshape(bsz, q_len, -1)
```

## 2. Transformers Version Mismatch
**Repo Requirement:** `transformers==4.48.2`
**Current Environment:** `transformers==4.56.0`

This significant version gap caused two major breaking changes:

### A. Missing `generate` Method
**Issue:**
Starting from `transformers` v4.50.0, the `PreTrainedModel` class no longer inherits from `GenerationMixin`. This means models loaded via `AutoModelForCausalLM` (or custom classes inheriting from `PreTrainedModel`) do not automatically have the `.generate()` method available.

**Error:**
`AttributeError: 'Emu3ForCausalLM' object has no attribute 'generate'`

**Fix:**
In `nodes.py`, we detect if the method is missing and dynamically inject the `GenerationMixin` into the model's class bases at runtime:
```python
from transformers.generation.utils import GenerationMixin
cls = model.__class__
if GenerationMixin not in cls.__bases__:
    cls.__bases__ = (GenerationMixin,) + cls.__bases__
```
We also ensure `model.generation_config` is explicitly set to avoid `NoneType` errors during generation preparation.

### B. `DynamicCache` API Changes
**Issue:**
The `DynamicCache` class in newer `transformers` versions removed or renamed several attributes that the Emu 3.5 code relied on:
1. `seen_tokens` -> Removed (use `get_seq_length()`)
2. `get_max_length()` -> Removed (use `max_cache_len` property or handle None)
3. `get_usable_length()` -> Removed in some contexts.

**Error:**
`AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`
`AttributeError: 'DynamicCache' object has no attribute 'get_max_length'`

**Fix:**
Patched `src/emu3p5/modeling_emu3.py` to add safe fallbacks:
```python
# Example Fix
if hasattr(past_key_values, "seen_tokens"):
    past_length = past_key_values.seen_tokens
else:
    past_length = cache_length # derived from get_seq_length()
```

## Summary
The combination of an upstream logic bug in the fallback attention path (SDPA) and breaking changes in the `transformers` library required patching both the node logic and the vendored model code. These fixes ensure the model runs correctly on standard ComfyUI installations without downgrading core libraries.
