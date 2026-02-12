"""Bridge to bio2token encode/decode functionality."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from bio2token.models.encoder import Encoder, EncoderConfig
from bio2token.models.decoder import Decoder, DecoderConfig
from bio2token.layers.fsq import FSQConfig
from bio2token.layers.mamba import MambaConfig
from bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks
from bio2token.data.utils.molecule_conventions import (
    ABBRS,
    BB_ATOMS_AA,
    SC_ATOMS_AA,
    AMINO_ACID_ABBRS,
)
from bio2token.data.utils.tokens import BB_CLASS, C_REF_CLASS, SC_CLASS, PAD_CLASS
from bio2token.utils.registration import Registration


@dataclass
class EncodeResult:
    token_ids: list[int]
    token_string: str
    num_tokens: int
    gt_pdb_string: str
    atom_names: list[str]
    residue_types: list[str]
    residue_ids: list[int]
    residue_names: list[str]
    token_classes: list[int]


@dataclass
class DecodeResult:
    pdb_string: str
    num_atoms: int


def _find_checkpoint() -> str:
    """Find the bio2token checkpoint file."""
    # 1. Environment variable
    env_path = os.environ.get("BIO2TOKEN_CHECKPOINT")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. bio2token repo checkout (sibling directory)
    bio2token_repo = Path("/home/ubuntu/bio2token")
    ckpt_dir = bio2token_repo / "checkpoints" / "bio2token" / "bio2token_pretrained"
    if ckpt_dir.is_dir():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # Prefer "best" checkpoint
            best = [c for c in ckpts if "best" in c.name]
            return str(best[0] if best else ckpts[0])

    # 3. Local checkpoints directory
    local_ckpt = Path("checkpoints")
    if local_ckpt.is_dir():
        ckpts = list(local_ckpt.glob("**/*.ckpt"))
        if ckpts:
            best = [c for c in ckpts if "best" in c.name]
            return str(best[0] if best else ckpts[0])

    raise FileNotFoundError(
        "Could not find bio2token checkpoint. Set BIO2TOKEN_CHECKPOINT env var "
        "or place checkpoints in /home/ubuntu/bio2token/checkpoints/"
    )


def _build_encoder() -> Encoder:
    """Build encoder with bio2token pretrained architecture config."""
    config = EncoderConfig(
        encoder_type="mamba",
        use_quantizer=True,
        quantizer_type="fsq",
    )
    config.encoder.mamba = MambaConfig(
        d_input=3,
        d_output=128,
        d_model=128,
        n_layer=4,
        bidirectional=True,
    )
    config.quantizer.fsq = FSQConfig(
        levels=[4, 4, 4, 4, 4, 4],
        d_input=128,
    )
    return Encoder(config)


def _build_decoder() -> Decoder:
    """Build decoder with bio2token pretrained architecture config."""
    config = DecoderConfig(decoder_type="mamba")
    config.decoder.mamba = MambaConfig(
        d_input=128,
        d_output=3,
        d_model=128,
        n_layer=6,
        bidirectional=True,
    )
    return Decoder(config)


def _coords_to_pdb_string(
    coords: np.ndarray,
    atom_names: list[str],
    residue_types: list[str],
    residue_ids: list[int],
) -> str:
    """Generate a PDB-format string from coordinates and metadata."""
    lines = []
    atom_num = 1
    current_res_num = 1
    prev_res_id = residue_ids[0] if len(residue_ids) > 0 else 0

    for coord, atom, res, res_id in zip(coords, atom_names, residue_types, residue_ids):
        if res_id != prev_res_id:
            current_res_num += 1
            prev_res_id = res_id

        # Standard PDB ATOM line format
        line = (
            f"ATOM  {atom_num:5d}  {atom:<3s} {res:3s} A{current_res_num:4d}"
            f"    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"  1.00  0.00           {atom[0]:>2s}"
        )
        lines.append(line)
        atom_num += 1

    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse_pdb_coords(pdb_string: str) -> np.ndarray:
    """Extract (N, 3) coordinate array from a PDB-format string."""
    coords = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def _kabsch_align(decoded: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align decoded coords to target coords via Kabsch (rigid body) alignment.

    Both inputs are (N, 3). Returns the aligned decoded coords.
    If lengths differ, aligns using the overlapping prefix.
    """
    n = min(len(decoded), len(target))
    if n < 3:
        return decoded

    reg = Registration()
    P = torch.tensor(decoded[:n], dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    Q = torch.tensor(target[:n], dtype=torch.float32).unsqueeze(0)   # (1, N, 3)

    aligned, _, _ = reg.transform(P, Q)
    result = aligned.squeeze(0).numpy()

    # If decoded was longer than target, transform the tail with the same rotation
    if len(decoded) > n:
        rot, tran = reg.get_transform(P, Q)
        full_P = torch.tensor(decoded, dtype=torch.float32).unsqueeze(0)
        full_aligned = reg.apply_transform(full_P, rot, tran)
        result = full_aligned.squeeze(0).numpy()

    return result


def _generate_default_metadata(num_tokens: int) -> tuple[list[str], list[str], list[int], list[str], list[int]]:
    """Generate default CA/ALA metadata when no real metadata is available.

    Each token maps to one ALA residue with a single CA atom (backbone-only).
    """
    atom_names = ["CA"] * num_tokens
    residue_types = ["ALA"] * num_tokens
    residue_ids = list(range(num_tokens))
    residue_names = ["AA_A"] * num_tokens
    token_classes = [C_REF_CLASS] * num_tokens
    return atom_names, residue_types, residue_ids, residue_names, token_classes


class Bio2TokenBridge:
    """Wraps bio2token encoder/decoder for programmatic use."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder: Encoder | None = None
        self.decoder: Decoder | None = None
        self.fsq = None
        self._loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load model weights from checkpoint."""
        ckpt_path = _find_checkpoint()
        print(f"Loading checkpoint from: {ckpt_path}")

        self.encoder = _build_encoder()
        self.decoder = _build_decoder()

        # Load checkpoint (Lightning format with "model." prefix)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        # Extract encoder weights
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model.encoder."):
                encoder_sd[k.replace("model.encoder.", "")] = v
        self.encoder.load_state_dict(encoder_sd)

        # Extract decoder weights
        decoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model.decoder."):
                decoder_sd[k.replace("model.decoder.", "")] = v
        self.decoder.load_state_dict(decoder_sd)

        self.encoder.eval().to(self.device)
        self.decoder.eval().to(self.device)

        # Keep a reference to the FSQ quantizer for indices_to_codes
        self.fsq = self.encoder.quantizer

        self._loaded = True
        print("Model loaded successfully.")

    def encode_pdb(self, pdb_content: str, filename: str = "upload.pdb") -> EncodeResult:
        """Encode a PDB file content string into bio2token tokens."""
        # Write to temp file since pdb_2_dict expects a path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            tmp_path = f.name

        try:
            pdb_dict = pdb_2_dict(tmp_path)
        finally:
            os.unlink(tmp_path)

        structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
            pdb_dict["seq"],
            pdb_dict["res_types"],
            pdb_dict["coords_groundtruth"],
            pdb_dict["atom_names"],
            pdb_dict["res_atom_start"],
            pdb_dict["res_atom_end"],
        )

        # Build batch — filter out unknown atoms
        batch = {
            "structure": torch.tensor(structure).float(),
            "unknown_structure": torch.tensor(unknown_structure).bool(),
            "residue_ids": torch.tensor(residue_ids).long(),
            "token_class": torch.tensor(token_class).long(),
        }

        # Keep track of known mask before filtering
        known_mask = ~batch["unknown_structure"]

        # Filter known-structure atoms and their metadata
        batch = {k: v[known_mask] for k, v in batch.items()}
        atom_names_known = [n for n, m in zip(atom_names_reordered, known_mask.numpy()) if m]
        residue_name_known = [n for n, m in zip(residue_name, known_mask.numpy()) if m]

        # Compute masks and add batch dimension
        batch = compute_masks(batch, structure_track=True)
        batch = {k: v[None].to(self.device) for k, v in batch.items()}

        # Generate ground-truth PDB string
        gt_coords = batch["structure"][0].detach().cpu().numpy()
        res_types_for_pdb = []
        for rn in residue_name_known:
            parts = rn.split("_")
            res_types_for_pdb.append(ABBRS[parts[0]][parts[1]])
        res_ids_for_pdb = batch["residue_ids"][0].detach().cpu().numpy().tolist()
        gt_pdb_string = _coords_to_pdb_string(gt_coords, atom_names_known, res_types_for_pdb, res_ids_for_pdb)

        # Encode
        with torch.no_grad():
            batch = self.encoder(batch)

        # Extract token indices (exclude padding)
        eos_pad_mask = batch["eos_pad_mask"][0].cpu()
        indices = batch["indices"][0].cpu()
        valid_indices = indices[~eos_pad_mask].numpy().tolist()

        # Build token string
        token_string = " ".join(f"<b{idx}>" for idx in valid_indices)

        # Extract token_class for valid positions
        valid_token_classes = batch["token_class"][0][~eos_pad_mask].cpu().numpy().tolist()

        return EncodeResult(
            token_ids=valid_indices,
            token_string=token_string,
            num_tokens=len(valid_indices),
            gt_pdb_string=gt_pdb_string,
            atom_names=atom_names_known,
            residue_types=res_types_for_pdb,
            residue_ids=res_ids_for_pdb,
            residue_names=residue_name_known,
            token_classes=valid_token_classes,
        )

    def decode_tokens(
        self,
        token_ids: list[int],
        atom_names: list[str] | None = None,
        residue_types: list[str] | None = None,
        residue_ids: list[int] | None = None,
        gt_pdb_string: str | None = None,
    ) -> DecodeResult:
        """Decode bio2token token IDs back to a PDB string.

        If gt_pdb_string is provided, the decoded coordinates are Kabsch-aligned
        to the ground-truth structure for proper visual overlay.
        """
        num_tokens = len(token_ids)

        # Use provided metadata or fall back to defaults
        if atom_names is None or residue_types is None or residue_ids is None:
            atom_names, residue_types, residue_ids, _, _ = _generate_default_metadata(num_tokens)

        # Convert token IDs to encoding vectors via FSQ
        indices = torch.tensor(token_ids, dtype=torch.int32).to(self.device)
        indices = indices[None, :]  # Add batch dimension: (1, seq_len)

        with torch.no_grad():
            encoding = self.fsq.indices_to_codes(indices, project_out=True)

            # Build batch for decoder
            batch = {
                "encoding": encoding,
                "eos_pad_mask": torch.zeros(1, num_tokens, dtype=torch.bool, device=self.device),
            }
            batch = self.decoder(batch)

        # Extract decoded coordinates
        coords = batch["decoding"][0].cpu().numpy()

        # Kabsch-align to ground truth if available
        if gt_pdb_string:
            gt_coords = _parse_pdb_coords(gt_pdb_string)
            if len(gt_coords) >= 3:
                coords = _kabsch_align(coords, gt_coords)

        # Generate PDB string
        pdb_string = _coords_to_pdb_string(coords, atom_names, residue_types, residue_ids)

        return DecodeResult(pdb_string=pdb_string, num_atoms=num_tokens)
