"""
Simple script to download Emu3.5-NF4 weights from HuggingFace Hub
"""

from huggingface_hub import snapshot_download
import os

# Target directory
target_dir = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\emu35\emu3.5-Image-nf4"

# Create directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading Emu3.5-NF4 weights to: {target_dir}")
print("This may take a while depending on your connection speed...")

# Download the entire repository
snapshot_download(
    repo_id="wikeeyang/Emu35-NF4",
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # Use actual files, not symlinks
    resume_download=True,  # Resume if interrupted
)

print(f"\nDownload complete! Files saved to: {target_dir}")
