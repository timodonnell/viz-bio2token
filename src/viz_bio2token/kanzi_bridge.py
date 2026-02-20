"""Bridge to Kanzi (flow-based protein structure tokenizer) encode/decode functionality."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from viz_bio2token.bio2token_bridge import (
    DecodeResult,
    EncodeResult,
    _kabsch_align,
    _parse_pdb_coords,
)
from viz_bio2token.apt_bridge import (
    _ca_coords_to_pdb_string,
    _extract_ca_coords_and_metadata,
)

# flex_attention patch applied in viz_bio2token.__init__
from kanzi.models import DAE, DAEConfig

MAX_KANZI_RESIDUES = 256


def _load_kanzi_without_gpt(ckpt_path: str) -> DAE:
    """Load Kanzi DAE, disabling the GPT prior component.

    The GPT prior requires flex_attention (PyTorch >= 2.5) but is only used
    for unconditional generation, not for encode/decode.  We disable it so
    the model loads on PyTorch 2.4.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = DAEConfig(**ckpt["model_cfg"])
    cfg.gpt_prior = False  # skip GPT init (avoids flex_attention)

    model = DAE(cfg)

    # Load weights, ignoring missing GPT keys
    model.load_state_dict(ckpt["model"], strict=False)
    return model


def _find_kanzi_checkpoint() -> str:
    """Find the Kanzi checkpoint file."""
    # 1. Environment variable
    env_path = os.environ.get("KANZI_CHECKPOINT")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Kanzi repo checkout (sibling directory)
    kanzi_dir = Path("/home/ubuntu/kanzi/checkpoints")
    if kanzi_dir.is_dir():
        ckpts = list(kanzi_dir.glob("**/*.ckpt")) + list(kanzi_dir.glob("**/*.pt"))
        if ckpts:
            return str(ckpts[0])

    # 3. Local checkpoints directory
    local_ckpt = Path("checkpoints")
    if local_ckpt.is_dir():
        ckpts = (
            list(local_ckpt.glob("**/kanzi*/*.ckpt")) + list(local_ckpt.glob("**/kanzi*.ckpt"))
            + list(local_ckpt.glob("**/kanzi*/*.pt")) + list(local_ckpt.glob("**/kanzi*.pt"))
        )
        if ckpts:
            return str(ckpts[0])

    raise FileNotFoundError(
        "Could not find Kanzi checkpoint. Set KANZI_CHECKPOINT env var "
        "or place checkpoints in /home/ubuntu/kanzi/checkpoints/"
    )


class KanziBridge:
    """Wraps Kanzi DAE tokenizer for encode/decode."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load Kanzi DAE model from checkpoint."""
        ckpt_path = _find_kanzi_checkpoint()
        print(f"Loading Kanzi model from: {ckpt_path}")
        self.model = _load_kanzi_without_gpt(ckpt_path)
        self.model.eval().to(self.device)
        self._loaded = True
        print("Kanzi model loaded successfully.")

    def encode_pdb(self, pdb_content: str, filename: str = "upload.pdb") -> EncodeResult:
        """Encode a PDB file into Kanzi tokens. Uses only CA atoms, 1 token per residue."""
        ca_coords, res_names, chains, res_nums = _extract_ca_coords_and_metadata(pdb_content)

        if len(ca_coords) == 0:
            raise ValueError("No CA atoms found in PDB")

        # Truncate to max supported sequence length
        if len(ca_coords) > MAX_KANZI_RESIDUES:
            ca_coords = ca_coords[:MAX_KANZI_RESIDUES]
            res_names = res_names[:MAX_KANZI_RESIDUES]
            chains = chains[:MAX_KANZI_RESIDUES]
            res_nums = res_nums[:MAX_KANZI_RESIDUES]

        num_residues = len(ca_coords)

        # Build ground-truth PDB from original CA coords
        gt_pdb_string = _ca_coords_to_pdb_string(ca_coords, res_names, chains, res_nums)

        # Kanzi handles centering+scaling internally with preprocess=True
        x_BLD = torch.tensor(ca_coords, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _s_BLD, _c_BLD, idx_BL = self.model.encode(x_BLD, preprocess=True)

        token_ids = idx_BL[0].cpu().tolist()
        token_string = " ".join(str(t) for t in token_ids)

        return EncodeResult(
            token_ids=token_ids,
            token_string=token_string,
            num_tokens=len(token_ids),
            gt_pdb_string=gt_pdb_string,
            num_residues=num_residues,
            max_tokens=num_residues,  # 1:1 mapping
        )

    def decode_tokens(
        self,
        token_ids: list[int],
        n_steps: int = 100,
        cfg_weight: float = 1.0,
        gt_pdb_string: str | None = None,
    ) -> DecodeResult:
        """Decode Kanzi token IDs back to a CA-only PDB string."""
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            coords = self.model.decode(idx, n_steps=n_steps, cfg_weight=cfg_weight)

        # Scale back to Angstroms
        coords_np = coords[0].cpu().numpy() * 10.0

        # Kabsch-align to ground truth if available
        if gt_pdb_string:
            gt_coords = _parse_pdb_coords(gt_pdb_string)
            if len(gt_coords) >= 3:
                coords_np = _kabsch_align(coords_np, gt_coords)

        pdb_string = _ca_coords_to_pdb_string(coords_np)

        return DecodeResult(pdb_string=pdb_string, num_atoms=len(coords_np))
