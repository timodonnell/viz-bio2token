"""Bridge to APT (Adaptive Protein Tokenization) encode/decode functionality."""

from __future__ import annotations

import sys
import types
from typing import Optional

import numpy as np
import torch

from viz_bio2token.bio2token_bridge import (
    DecodeResult,
    EncodeResult,
    _coords_to_pdb_string,
    _kabsch_align,
    _parse_pdb_coords,
)


def _patch_flex_attention():
    """Ensure APT can import on PyTorch < 2.5 where flex_attention is missing.

    APT unconditionally imports from torch.nn.attention.flex_attention at
    module level.  On PyTorch 2.4 that subpackage doesn't exist, so we
    inject a tiny stub *before* importing any APT code.
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


# Patch before any APT imports
_patch_flex_attention()

from apt.models import APTTokenizer  # noqa: E402


MAX_APT_TOKENS = 128


def _extract_ca_coords_and_metadata(pdb_string: str):
    """Extract CA atom coordinates and per-residue metadata from a PDB string.

    Returns (coords, residue_names, chain_ids, residue_numbers) where
    coords is (N, 3) float32 array for CA atoms only.
    """
    coords = []
    residue_names = []
    chain_ids = []
    residue_numbers = []

    for line in pdb_string.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append([x, y, z])
        residue_names.append(line[17:20].strip())
        chain_ids.append(line[21:22].strip() or "A")
        residue_numbers.append(int(line[22:26].strip()))

    return (
        np.array(coords, dtype=np.float32),
        residue_names,
        chain_ids,
        residue_numbers,
    )


def _ca_coords_to_pdb_string(
    coords: np.ndarray,
    residue_names: list[str] | None = None,
    chain_ids: list[str] | None = None,
    residue_numbers: list[int] | None = None,
) -> str:
    """Generate a CA-only PDB string from coordinates."""
    n = len(coords)
    if residue_names is None:
        residue_names = ["ALA"] * n
    if chain_ids is None:
        chain_ids = ["A"] * n
    if residue_numbers is None:
        residue_numbers = list(range(1, n + 1))

    atom_names = ["CA"] * n
    residue_ids = list(range(n))  # 0-based for _coords_to_pdb_string
    return _coords_to_pdb_string(coords, atom_names, residue_names, residue_ids, chain_ids)


class APTBridge:
    """Wraps APT tokenizer for encode/decode."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self._loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load APT tokenizer (auto-downloads from HuggingFace Hub)."""
        print("Loading APT tokenizer...")
        self.tokenizer = APTTokenizer.from_pretrained()
        self.tokenizer.eval().to(self.device)
        self._loaded = True
        print("APT tokenizer loaded successfully.")

    def encode_pdb(self, pdb_content: str, filename: str = "upload.pdb") -> EncodeResult:
        """Encode a PDB file into APT tokens. Uses only CA atoms."""
        ca_coords, res_names, chains, res_nums = _extract_ca_coords_and_metadata(pdb_content)

        if len(ca_coords) == 0:
            raise ValueError("No CA atoms found in PDB")

        num_residues = len(ca_coords)

        # Build ground-truth PDB from original CA coords
        gt_pdb_string = _ca_coords_to_pdb_string(ca_coords, res_names, chains, res_nums)

        # Preprocess: zero-center and scale
        centered = ca_coords - ca_coords.mean(axis=0, keepdims=True)
        scaled = centered / 10.0

        x_BLD = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _s_BLD, _c_BLD, idx_BL = self.tokenizer.encode(x_BLD)

        # Truncate to max tokens
        idx_BL = idx_BL[:, :MAX_APT_TOKENS]
        token_ids = idx_BL[0].cpu().tolist()
        token_string = " ".join(str(t) for t in token_ids)

        return EncodeResult(
            token_ids=token_ids,
            token_string=token_string,
            num_tokens=len(token_ids),
            gt_pdb_string=gt_pdb_string,
            num_residues=num_residues,
            max_tokens=MAX_APT_TOKENS,
        )

    def decode_tokens(
        self,
        token_ids: list[int],
        num_residues: int | None = None,
        n_steps: int = 100,
        gt_pdb_string: str | None = None,
    ) -> DecodeResult:
        """Decode APT token IDs back to a CA-only PDB string."""
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        true_length = num_residues

        with torch.no_grad():
            coords = self.tokenizer.decode(idx, true_length=true_length, n_steps=n_steps)

        # Scale back to Angstroms
        coords_np = coords[0].cpu().numpy() * 10.0

        # Kabsch-align to ground truth if available
        if gt_pdb_string:
            gt_coords = _parse_pdb_coords(gt_pdb_string)
            if len(gt_coords) >= 3:
                coords_np = _kabsch_align(coords_np, gt_coords)

        pdb_string = _ca_coords_to_pdb_string(coords_np)

        return DecodeResult(pdb_string=pdb_string, num_atoms=len(coords_np))
