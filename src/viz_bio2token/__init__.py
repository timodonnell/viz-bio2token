"""viz-bio2token package.

Patches torch.nn.attention.flex_attention before any bridge modules are imported,
so both APT and Kanzi can load on PyTorch < 2.5.
"""

import sys
import types


def _patch_flex_attention():
    """Ensure APT/Kanzi can import on PyTorch < 2.5 where flex_attention is missing.

    Both packages unconditionally import from torch.nn.attention.flex_attention at
    module level.  On PyTorch 2.4 that subpackage doesn't exist, so we inject a
    tiny stub *before* importing any APT/Kanzi code.
    """
    target = "torch.nn.attention.flex_attention"
    if target in sys.modules:
        return  # already available (PyTorch >= 2.5) or already patched

    try:
        __import__(target)
        return  # real module exists
    except (ImportError, ModuleNotFoundError):
        pass

    stub = types.ModuleType(target)
    stub.flex_attention = None
    stub.BlockMask = None
    stub.create_block_mask = None
    stub.and_masks = None
    sys.modules[target] = stub

    # Also ensure the parent chain exists
    parent = "torch.nn.attention"
    if parent not in sys.modules:
        parent_mod = types.ModuleType(parent)
        parent_mod.flex_attention = stub
        sys.modules[parent] = parent_mod


_patch_flex_attention()
