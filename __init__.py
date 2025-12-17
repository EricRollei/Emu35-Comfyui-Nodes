"""
Emu3.5 ComfyUI Custom Nodes
Supports T2I, X2I (image editing), Interleaved Generation, and VQA.
"""

# Import both old and new nodes for compatibility
from .nodes import NODE_CLASS_MAPPINGS as OLD_NODES, NODE_DISPLAY_NAME_MAPPINGS as OLD_NAMES
from .nodes_v2 import NODE_CLASS_MAPPINGS as NEW_NODES, NODE_DISPLAY_NAME_MAPPINGS as NEW_NAMES

# Merge node mappings (new nodes take precedence on conflicts)
NODE_CLASS_MAPPINGS = {**OLD_NODES, **NEW_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**OLD_NAMES, **NEW_NAMES}

# Web directory for any JS extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
