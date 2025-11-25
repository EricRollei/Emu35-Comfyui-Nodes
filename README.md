# Emu3.5 ComfyUI Nodes

**Experimental ComfyUI integration for BAAI's Emu3.5 multimodal models**

⚠️ **STATUS: RESEARCH/EXPERIMENTAL** - Emu3.5-Image model is incomplete and not production-ready. See [Known Issues](#known-issues) below.

## Overview

This repository provides ComfyUI custom nodes for running BAAI's Emu3.5 models for text-to-image generation and multimodal understanding. 

**Models Supported:**
- Emu3.5-Image (Text-to-Image) - ⚠️ Currently produces low-quality outputs
- Emu3.5-Base (Foundation model)
- Vision Tokenizer (VQ-VAE for image encoding/decoding)

## Credits & Attribution

This project is built upon and inspired by:

- **[BAAI Emu3.5](https://github.com/baaivision/Emu3.5)** - Original model and codebase
  - Paper: [Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/pdf/2510.26583)
  - Authors: Emu3.5 Team, Beijing Academy of Artificial Intelligence
  - License: Apache 2.0

- **[BAAI Emu3](https://github.com/baaivision/Emu3)** - Predecessor model (more stable for production use)
  - Paper: [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/pdf/2409.18869)

All model weights and architecture remain property of BAAI under Apache 2.0 license.

## Installation

### Prerequisites
- ComfyUI installed
- Python 3.10+
- CUDA-capable GPU with 24GB+ VRAM (FP16) or 16GB+ (NF4 quantization)
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

```bash
# From Hugging Face
huggingface-cli download BAAI/Emu3.5-Image --local-dir models/emu35/emu3.5-Image
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
        ├── emu3.5-Image/
        └── vision_tokenizer/
```

## Usage

### Nodes Available

#### 1. Emu35 Loader
Loads the Emu3.5 model, tokenizer, and VQ model.

**Inputs:**
- `model_name`: Select model folder (e.g., "emu3.5-Image")
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
- `cfg_scale`: Classifier-free guidance (1.0-20.0, default 3.0)
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

## Known Issues

⚠️ **Critical**: The Emu3.5-Image model currently produces **low-quality/corrupted outputs**. Investigation reveals:

1. **Model generates malformed token sequences**: Inconsistent row lengths, missing end tokens
2. **Incomplete implementation**: BAAI's roadmap shows "Advanced image decoder" and "Diffusion adaptation" are TODO
3. **Research preview status**: Model appears to be an incomplete research checkpoint

**Evidence:**
- Generated image token grids have irregular shapes (e.g., 525 tokens in one row, 4 in another)
- Missing EOI (End of Image) tokens in output
- Same issues occur with official `inference.py` script

### Recommended Alternatives

For production use, consider:
- **[BAAI/Emu3-Gen](https://huggingface.co/BAAI/Emu3-Gen)** - Stable predecessor model with working T2I
- **Stable Diffusion** / **FLUX** - More mature text-to-image solutions
- **Wait for Emu3.5 completion** - Monitor BAAI's repo for updates

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
**Solution**: This warning is expected. The code automatically adds `GenerationMixin` to fix this.

### Issue: "Tokenizer crashes on loading"
**Solution**: The patched tokenizer synthesizes missing visual tokens automatically.

### Issue: "Out of memory"
**Solution**: 
- Use `nf4` precision (requires ~16GB VRAM)
- Reduce image dimensions
- Enable `--lowvram` in ComfyUI launch args

### Issue: "Images are garbage/static"
**Status**: Known issue with Emu3.5-Image checkpoint. See [Known Issues](#known-issues).

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
