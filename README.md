# Emu3.5 ComfyUI Nodes

**ComfyUI integration for BAAI's Emu3.5 multimodal models**

✅ **STATUS: WORKING** - Text-to-image generation verified on December 7, 2025

![Example Output](assets/Emu35Image_2025-12-07_15-06-02_000.png)

## Overview

This repository provides ComfyUI custom nodes for running BAAI's Emu3.5 models for text-to-image generation and multimodal understanding. 

**Models Supported:**
- Emu3.5-Image (Text-to-Image) - ✅ Working
- Emu3.5-Base (Foundation model)
- Vision Tokenizer (VQ-VAE for image encoding/decoding)

## Credits & Attribution

This project is built upon and inspired by:

- **[BAAI Emu3.5](https://github.com/baaivision/Emu3.5)** - Original model and codebase
  - Paper: [Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/pdf/2510.26583)
  - Authors: Emu3.5 Team, Beijing Academy of Artificial Intelligence
  - License: Apache 2.0

- **[BAAI Emu3](https://github.com/baaivision/Emu3)** - Predecessor model
  - Paper: [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/pdf/2409.18869)

All model weights and architecture remain property of BAAI under Apache 2.0 license.

## Installation

### Prerequisites
- ComfyUI installed
- Python 3.10+
- CUDA-capable GPU with 24GB+ VRAM (BF16) or 16GB+ (NF4 quantization)
- 100GB+ disk space for model weights

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Emu35-Comfyui-Nodes.git emu35
cd emu35

# Clone the official Emu3.5 repository as a submodule
git clone https://github.com/baaivision/Emu3.5.git Emu3_5_repo

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Manual Download

1. Download this repository
2. Extract to `ComfyUI/custom_nodes/emu35/`
3. Download [Emu3.5 repo](https://github.com/baaivision/Emu3.5) and place in `emu35/Emu3_5_repo/`
4. Install requirements: `pip install -r requirements.txt`

### Download Model Weights

Place models in `ComfyUI/models/emu35/`:

**Option A: Full BF16 (Recommended - 48GB+ VRAM)**
```bash
# Full precision - best quality
huggingface-cli download BAAI/Emu3.5-Image --local-dir models/emu35/Emu3.5-Image
huggingface-cli download BAAI/Emu3.5-VisionTokenizer --local-dir models/emu35/vision_tokenizer
```

**Option B: NF4 Quantized (24GB VRAM)**
```bash
# NF4 quantized version - works with 24-32GB VRAM
huggingface-cli download wikeeyang/Emu35-Image-NF4 --local-dir models/emu35/Emu3.5-Image-NF4
huggingface-cli download BAAI/Emu3.5-VisionTokenizer --local-dir models/emu35/vision_tokenizer
```

**Directory structure:**
```
ComfyUI/
├── custom_nodes/
│   └── emu35/
│       ├── nodes.py
│       ├── __init__.py
│       ├── patched_tokenization_emu3.py
│       └── Emu3_5_repo/  (official repo)
└── models/
    └── emu35/
        ├── Emu3.5-Image/  (or Emu3.5-Image-NF4/)
        └── vision_tokenizer/
```

## Usage

### Nodes Available

#### 1. Emu35 Loader
Loads the Emu3.5 model, tokenizer, and VQ model.

**Inputs:**
- `model_name`: Select model folder (e.g., "Emu3.5-Image")
- `vq_model_name`: Vision tokenizer folder (usually "vision_tokenizer")
- `precision`: bf16 (default), fp16, fp32, or nf4 (quantized)

**Outputs:**
- `model`: Loaded Emu3.5 model
- `tokenizer`: Tokenizer
- `vq_model`: Vision tokenizer for image encoding/decoding

#### 2. Emu35 Sampler
Generates images from text prompts.

**Inputs:**
- `model`, `tokenizer`, `vq_model`: From loader
- `prompt`: Text description of desired image
- `negative_prompt`: What to avoid in generation
- `width`/`height`: Output dimensions (must be multiples of 64, range 256-2048)
- `steps`: Generation steps (1-200, default 50)
- `cfg_scale`: Classifier-free guidance (1.0-20.0, default 5.0 recommended)
- `seed`: Random seed for reproducibility

**Outputs:**
- `image`: Generated image
- `text_response`: Any text generated alongside the image

### Example Workflow

```
[Emu35 Loader] → [Emu35 Sampler] → [Preview Image]
                      ↑
              [Text Prompt: "a red apple on a table"]
```

## Status Update (December 2025)

✅ **WORKING** - Text-to-image generation is now fully functional!

### Key Fixes Applied

1. **SDPA Attention Bug on Blackwell GPUs**: SDPA attention produces noise/garbage on Blackwell architecture (sm_120) with CUDA 12.8. Fixed by using `attn_implementation="eager"`.

2. **Transformers 4.57+ Compatibility**: The official Emu3.5 code was written for transformers 4.48. We've patched `modeling_emu3.py` to work with transformers 4.50+:
   - Added `GenerationMixin` inheritance for `generate()` method
   - Fixed `DynamicCache` API changes (`get_usable_length` → `get_seq_length`)
   - Fixed attention mask size mismatches with proper `-inf` padding

3. **Attention Mask Handling**: Fixed critical bugs in mask padding:
   - Use `-inf` (not zeros) to mask non-existent positions
   - Keep FIRST tokens when trimming (history/prompt), not LAST

4. **Prompt Format**: Using official chat-style template:
```
<|extra_203|>You are a helpful assistant. USER: {prompt} ASSISTANT:
```

### Tested Configuration

| Component | Version/Setting |
|-----------|-----------------|
| GPU | NVIDIA RTX 6000 Pro (Blackwell) |
| CUDA | 12.8 |
| PyTorch | 2.7+ |
| Transformers | 4.57.1 |
| Attention | `eager` (not SDPA) |
| Precision | BF16 |

## Known Issues

1. **VRAM Requirements**: Full BF16 model needs 48GB+, NF4 quantized needs 24-32GB

2. **Generation Speed**: Autoregressive generation is slower than diffusion models (~10-15 min for 512x512)

3. **Blackwell GPU + SDPA**: SDPA attention produces corrupted outputs on Blackwell GPUs. Use `eager` attention (automatically set).

4. **NF4 Quantization**: The `lm_head` must NOT be quantized or outputs will be garbage. Our loader verifies this.

### GPU Compatibility

| GPU Architecture | SDPA | Eager | Recommended |
|-----------------|------|-------|-------------|
| Ampere (sm_80) | ✅ | ✅ | SDPA |
| Ada Lovelace (sm_89) | ✅ | ✅ | SDPA |
| Blackwell (sm_120) | ❌ | ✅ | **Eager** |

## Technical Details

### Architecture
- **Base**: Llama-style transformer (8B parameters)
- **Training**: 10T+ multimodal tokens from video frames + transcripts
- **Image Tokenization**: VQ-VAE (IBQ) with 262,144 codebook size
- **Visual Tokens**: Token IDs 151855-413998 (262,144 discrete codes)
- **Resolution**: Supports up to 2048x2048 (128x128 latents)

### Special Token IDs
```python
BOS = 151849  # <|extra_203|> Begin generation
EOS = 151850  # <|extra_204|> End generation  
IMG = 151851  # <|image|>
BOI = 151852  # <|begin_of_image|>
EOI = 151853  # <|end_of_image|>
EOL = 151846  # <|extra_200|> End of line
VISUAL_START = 151855  # First visual token
```

### Prompt Format
```
<|extra_203|>{prompt}<|extra_204|>{H}*{W}<|image|>
```
Where H, W are latent dimensions (height/16, width/16).

## Troubleshooting

### Issue: "Model has no 'generate' method"
**Solution**: This is automatically fixed. The patched `modeling_emu3.py` adds `GenerationMixin` inheritance.

### Issue: "Tokenizer crashes on loading"
**Solution**: The patched tokenizer handles missing visual tokens automatically.

### Issue: "Out of memory"
**Solution**: 
- Use `nf4` precision (requires ~24GB VRAM)
- Reduce image dimensions (512x512 instead of 1024x1024)
- Enable `--lowvram` in ComfyUI launch args

### Issue: "Images are noise/garbage"
**Solution**: 
- If on Blackwell GPU: Ensure `attn_implementation="eager"` (automatic in our loader)
- If NF4 quantized: Verify `lm_head` is NOT quantized (check console output)
- Verify model weights with `python verify_hashes.py` (if available)

### Issue: "Attention mask size mismatch"
**Solution**: Our patched `modeling_emu3.py` handles transformers 4.57+ cache API changes. If you still see this error, delete `__pycache__` folders and restart.

## Development

### Files Overview
- `nodes.py`: Main ComfyUI node implementations
- `patched_tokenization_emu3.py`: Fixed tokenizer with synthesized visual tokens
- `__init__.py`: ComfyUI registration
- `test_*.py`: Diagnostic scripts for testing model behavior

### Testing
```bash
# Test basic generation (no CFG)
python test_minimal.py

# Test with logits processor
python test_with_processor.py

# Check token ID mappings
python check_special_tokens.py
```

## Contributing

This is an experimental project. Contributions welcome, especially:
- Fixes for token generation issues
- Alternative decoding strategies
- Better prompt engineering
- Documentation improvements

## License

This integration code: MIT License

Emu3.5 models and original code: Apache 2.0 (BAAI)

## Citation

If you use Emu3.5 in your research, please cite:

```bibtex
@article{emu3.5,
  title={Emu3.5: Native Multimodal Models are World Learners},
  author={Emu3.5 Team},
  journal={arXiv preprint arXiv:2510.26583},
  year={2024}
}

@article{emu3,
  title={Emu3: Next-Token Prediction is All You Need},
  author={Wang, Xinlong and others},
  journal={arXiv preprint arXiv:2409.18869},
  year={2024}
}
```

## Links

- **Emu3.5 Project**: https://emu.world/
- **Paper**: https://arxiv.org/pdf/2510.26583
- **Official Code**: https://github.com/baaivision/Emu3.5
- **Model Weights**: https://huggingface.co/BAAI/Emu3.5-Image
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

## Acknowledgments

Special thanks to:
- **BAAI Team** for releasing Emu3.5 and maintaining open-source models
- **ComfyUI Community** for the excellent node framework
- **Hugging Face** for model hosting infrastructure

---

**Disclaimer**: This is an unofficial, experimental integration. For official Emu3.5 usage, refer to [BAAI's repository](https://github.com/baaivision/Emu3.5).

---

## 🎯 Emu3.5 Feature To-Do List

### HIGH PRIORITY - Officially Supported, High Value

#### 1. Image Editing / Variation

| Attribute | Status |
|-----------|--------|
| **Status** | Parameter exists in Emu35Sampler but not implemented |
| **Official Support** | ✅ Yes (in official repo) |
| **Value** | ⭐⭐⭐⭐⭐ (Very useful for workflows) |
| **Complexity** | 🔧🔧 Medium |

**Description:** Feed an input image to the model for editing/variation

**Implementation:**
- Encode input image with VQ-VAE
- Format as: `BOI -> resolution -> IMG -> visual tokens -> EOI`
- Prepend to prompt or use specific editing template
- Model generates modified/variation of input

**Use Cases:**
- "Make this image look like it's at sunset"
- "Add a hat to the person in this photo"
- Style transfer, variations, refinements

---

#### 2. Chain-of-Thought Prompting

| Attribute | Status |
|-----------|--------|
| **Status** | CoT extraction implemented, but no way to REQUEST CoT |
| **Official Support** | ✅ Yes (special tokens set) |
| **Value** | ⭐⭐⭐⭐ (Research, debugging, better results) |
| **Complexity** | 🔧 Easy |

**Description:** Ask model to show reasoning BEFORE generating image

**Implementation:**
- Add checkbox: "Enable Chain-of-Thought"
- Modify prompt to request reasoning: "Think step by step before generating the image. {prompt}"
- Or wrap prompt in CoT tokens

**Use Cases:**
- Understanding why model chose certain composition
- Better results through explicit reasoning
- Research and analysis

---

### MEDIUM PRIORITY - Officially Supported, Moderate Value

#### 3. Multiple Images in Single Generation

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ✅ Yes (multimodal_decode handles it) |
| **Value** | ⭐⭐⭐⭐ (Creative workflows) |
| **Complexity** | 🔧🔧🔧 Medium-High |

**Description:** Model generates multiple images in one pass

**Implementation:**
- Change RETURN_TYPES to support image arrays
- Parse all images from multimodal_decode
- Return list/batch of images

**Challenges:**
- ComfyUI node output typing (might need custom output type)
- UI complexity (how to display multiple images)

**Use Cases:**
- "Generate a before and after comparison"
- "Show 3 variations of this concept"
- Sequential storytelling

---

#### 4. Streaming Generation

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ✅ Yes (streaming_generate function) |
| **Value** | ⭐⭐⭐ (UX improvement) |
| **Complexity** | 🔧🔧🔧🔧 High |

**Description:** Show progressive generation (text/image tokens as they're generated)

**Implementation:**
- Use official `streaming_generate()` instead of `generate()`
- Implement callback/webhook to update UI
- Requires async/threading support

**Challenges:**
- ComfyUI architecture doesn't support streaming well
- Complex to implement properly
- May need custom UI components

**Use Cases:**
- Real-time feedback during long generations
- Early stopping if generation goes wrong
- Better UX for slow models

---

#### 5. Video Generation (Emu3.5-Video)

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented at all |
| **Official Support** | ✅ Yes (separate model: Emu3.5-Video) |
| **Value** | ⭐⭐⭐⭐⭐ (Huge feature) |
| **Complexity** | 🔧🔧🔧🔧🔧 Very High |

**Description:** Text-to-Video and Image-to-Video generation

**Implementation:**
- Create separate loader for Video model
- Implement temporal dimension handling
- Video tokenization/detokenization
- Frame generation and encoding

**Requirements:**
- Emu3.5-Video model (separate download)
- Video codec support
- Significantly more VRAM

**Use Cases:**
- Text-to-video: "A cat walking across a room"
- Image-to-video: Animate a still image
- Video editing/style transfer

---

### LOW PRIORITY - Quality of Life Improvements

#### 6. Batch Generation

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ⚠️ Partial (model supports batch, but examples don't use it) |
| **Value** | ⭐⭐⭐ (Efficiency) |
| **Complexity** | 🔧🔧 Medium |

**Description:** Generate multiple images in one pass (different prompts/seeds)

**Implementation:**
- Add "batch_size" parameter
- Process multiple prompts in parallel
- Return batched tensor

**Use Cases:**
- Generate 4 variations with different seeds
- Compare different prompts side-by-side

---

#### 7. Advanced Sampling Parameters

| Attribute | Status |
|-----------|--------|
| **Status** | Using defaults |
| **Official Support** | ✅ Yes (many parameters available) |
| **Value** | ⭐⭐ (Power users) |
| **Complexity** | 🔧 Easy |

**Description:** Expose more generation parameters to users

**Current:** `image_top_k`, `temperature` are hardcoded

**Could Add:**
- `text_top_k`, `text_top_p`, `text_temperature`
- `image_top_p`, `image_temperature`
- `use_differential_sampling` toggle
- `num_beams` for beam search

**Use Cases:**
- Fine-tuning generation quality
- Experimentation
- Reproducibility research

---

#### 8. Negative Prompting (Better Implementation)

| Attribute | Status |
|-----------|--------|
| **Status** | Basic implementation in Emu35Sampler |
| **Official Support** | ⚠️ Indirect (CFG supports it) |
| **Value** | ⭐⭐⭐ (Quality control) |
| **Complexity** | 🔧 Easy |

**Description:** Better negative prompt handling

**Current Issue:** Negative prompt uses same template as positive

**Could Improve:**
- Test if different negative templates work better
- Add "negative weight" parameter
- Research optimal negative prompting for Emu3.5

**Use Cases:**
- "No hands, no text, no watermarks"
- Fine control over what NOT to generate

---

#### 9. LoRA / Fine-tuning Support

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ❓ Unknown (likely possible, not documented) |
| **Value** | ⭐⭐⭐⭐⭐ (Customization) |
| **Complexity** | 🔧🔧🔧🔧 High |

**Description:** Load and apply LoRA weights for custom styles

**Implementation:**
- LoRA weight loading
- Merging with base model
- Multi-LoRA support

**Use Cases:**
- Custom art styles
- Specific character/object training
- Domain adaptation

---

#### 10. Model/Tokenizer Caching

| Attribute | Status |
|-----------|--------|
| **Status** | Loads from scratch each time |
| **Official Support** | N/A (implementation detail) |
| **Value** | ⭐⭐⭐ (Performance) |
| **Complexity** | 🔧 Easy |

**Description:** Cache loaded models between runs

**Implementation:**
- Global cache dict
- Check cache before loading
- LRU eviction for memory management

**Use Cases:**
- Faster workflow iterations
- Multiple workflows using same model

---

#### 11. Image Resolution Validation

| Attribute | Status |
|-----------|--------|
| **Status** | No validation |
| **Official Support** | N/A (safety feature) |
| **Value** | ⭐⭐ (UX) |
| **Complexity** | 🔧 Easy |

**Description:** Warn/prevent invalid resolutions

**Implementation:**
- Check width/height are multiples of 16
- Warn if too large (VRAM limits)
- Auto-adjust to nearest valid size

**Use Cases:**
- Prevent user errors
- Better error messages

---

#### 12. Prompt Templates Library

| Attribute | Status |
|-----------|--------|
| **Status** | Templates are hardcoded |
| **Official Support** | N/A (UX feature) |
| **Value** | ⭐⭐ (Convenience) |
| **Complexity** | 🔧 Easy |

**Description:** Preset templates for different use cases

**Templates:**
- Photography styles
- Art styles
- Specific domains (architecture, product, portrait)

**Use Cases:**
- Quick starts for new users
- Consistent results

---

### EXPERIMENTAL - Not Officially Documented

#### 13. Inpainting/Outpainting

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ❓ Unknown |
| **Value** | ⭐⭐⭐⭐⭐ (Very powerful) |
| **Complexity** | 🔧🔧🔧🔧🔧 Very High |

**Description:** Edit specific regions or extend images

**Would Require:** Research into whether Emu3.5 supports this

---

#### 14. ControlNet-style Conditioning

| Attribute | Status |
|-----------|--------|
| **Status** | Not implemented |
| **Official Support** | ❓ Unknown |
| **Value** | ⭐⭐⭐⭐⭐ (Precise control) |
| **Complexity** | 🔧🔧🔧🔧🔧 Very High |

**Description:** Condition generation on edge maps, depth, etc.

**Would Require:** Research and possibly model fine-tuning

---

### 📋 Recommended Implementation Order

#### Phase 1: Quick Wins (1-2 weeks)
- [ ] Chain-of-Thought prompting (add checkbox)
- [ ] Advanced sampling parameters (expose existing params)
- [ ] Model caching (performance boost)

#### Phase 2: High Value (2-4 weeks)
- [ ] Image editing/variation (use input_image parameter)
- [ ] Negative prompt improvements
- [ ] Batch generation

#### Phase 3: Major Features (1-2 months)
- [ ] Multiple images per generation
- [ ] Video generation (Emu3.5-Video model)

#### Phase 4: Advanced (Research required)
- [ ] Streaming generation
- [ ] LoRA support
- [ ] Inpainting/outpainting (if possible)

---

### 🎯 Top 3 Recommendations

1. **Image Editing** - High value, moderate effort, officially supported
2. **CoT Prompting** - Easy win, improves quality, unique feature
3. **Video Generation** - Huge feature, but requires separate model
