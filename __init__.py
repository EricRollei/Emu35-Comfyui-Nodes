"""
Emu3.5 ComfyUI Custom Nodes
Supports T2I, X2I (image editing), Interleaved Generation, and VQA.
"""

# =============================================================================
# CRITICAL: Patch DynamicCache BEFORE any other imports
# Transformers 4.50+ removed 'seen_tokens' and 'get_usable_length' from DynamicCache
# The Emu3 model code expects these attributes to exist
# This MUST be done before importing nodes.py or nodes_v2.py
# =============================================================================
try:
    from transformers.cache_utils import DynamicCache
    
    # Patch 1: Add get_usable_length if missing
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, input_seq_length, layer_idx=None):
            if layer_idx is not None:
                return self.get_seq_length(layer_idx)
            else:
                return self.get_seq_length()
        DynamicCache.get_usable_length = _get_usable_length
        print("[Emu35] Patched DynamicCache.get_usable_length")
    
    # Patch 2: Add seen_tokens property if missing
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        print("[Emu35] Patched DynamicCache.seen_tokens")
        
except ImportError as e:
    print(f"[Emu35] Warning: Could not patch DynamicCache: {e}")

# Import both old and new nodes for compatibility
from .nodes import NODE_CLASS_MAPPINGS as OLD_NODES, NODE_DISPLAY_NAME_MAPPINGS as OLD_NAMES
from .nodes_v2 import NODE_CLASS_MAPPINGS as NEW_NODES, NODE_DISPLAY_NAME_MAPPINGS as NEW_NAMES

# Merge node mappings (new nodes take precedence on conflicts)
NODE_CLASS_MAPPINGS = {**OLD_NODES, **NEW_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**OLD_NAMES, **NEW_NAMES}

# Web directory for any JS extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
