"""FastAPI application for viz-bio2token."""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from viz_bio2token.bio2token_bridge import Bio2TokenBridge
from viz_bio2token.pdb_utils import cif_to_pdb, is_cif_format

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


class TokenizerType(str, Enum):
    bio2token = "bio2token"
    apt = "apt"
    kanzi = "kanzi"


bio2token_bridge = Bio2TokenBridge()
apt_bridge = None  # initialized in lifespan
apt_available = False
kanzi_bridge = None  # initialized in lifespan
kanzi_available = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tokenizer models on startup."""
    global apt_bridge, apt_available, kanzi_bridge, kanzi_available

    print("Loading bio2token model...")
    bio2token_bridge.load()

    try:
        from viz_bio2token.apt_bridge import APTBridge
        apt_bridge = APTBridge()
        apt_bridge.load()
        apt_available = True
    except Exception as e:
        print(f"APT tokenizer not available: {e}")
        apt_available = False

    try:
        from viz_bio2token.kanzi_bridge import KanziBridge
        kanzi_bridge = KanziBridge()
        kanzi_bridge.load()
        kanzi_available = True
    except Exception as e:
        print(f"Kanzi tokenizer not available: {e}")
        kanzi_available = False

    yield


app = FastAPI(title="viz-bio2token", lifespan=lifespan)


# --- Request/Response models ---

class DecodeRequest(BaseModel):
    token_ids: list[int]
    tokenizer: TokenizerType = TokenizerType.bio2token
    # bio2token-specific
    atom_names: Optional[list[str]] = None
    residue_types: Optional[list[str]] = None
    residue_ids: Optional[list[int]] = None
    chain_ids: Optional[list[str]] = None
    gt_pdb_string: Optional[str] = None
    # APT-specific
    num_residues: Optional[int] = None
    n_steps: Optional[int] = 100
    # Kanzi-specific
    cfg_weight: Optional[float] = 1.0


class DecodeResponse(BaseModel):
    pdb_string: str
    num_atoms: int


class EncodeResponse(BaseModel):
    token_ids: list[int]
    token_string: str
    num_tokens: int
    gt_pdb_string: str
    tokenizer: str
    # bio2token-specific (optional for APT)
    atom_names: Optional[list[str]] = None
    residue_types: Optional[list[str]] = None
    residue_ids: Optional[list[int]] = None
    residue_names: Optional[list[str]] = None
    token_classes: Optional[list[int]] = None
    chain_ids: Optional[list[str]] = None
    # APT-specific
    num_residues: Optional[int] = None
    max_tokens: Optional[int] = None


class StatusResponse(BaseModel):
    bio2token_loaded: bool
    apt_loaded: bool
    kanzi_loaded: bool


# --- Helpers ---

def _get_bridge(tokenizer: TokenizerType):
    if tokenizer == TokenizerType.bio2token:
        if not bio2token_bridge.model_loaded:
            raise HTTPException(status_code=503, detail="bio2token model not loaded yet")
        return bio2token_bridge
    elif tokenizer == TokenizerType.apt:
        if not apt_available or apt_bridge is None or not apt_bridge.model_loaded:
            raise HTTPException(status_code=503, detail="APT model not available")
        return apt_bridge
    else:
        if not kanzi_available or kanzi_bridge is None or not kanzi_bridge.model_loaded:
            raise HTTPException(status_code=503, detail="Kanzi model not available")
        return kanzi_bridge


# --- API endpoints ---

@app.post("/api/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest):
    bridge = _get_bridge(req.tokenizer)

    # Validate token range
    for tid in req.token_ids:
        if tid < 0 or tid > 4095:
            raise HTTPException(status_code=400, detail=f"Token ID {tid} out of range [0, 4095]")

    if len(req.token_ids) == 0:
        raise HTTPException(status_code=400, detail="No token IDs provided")

    if req.tokenizer == TokenizerType.kanzi:
        result = bridge.decode_tokens(
            token_ids=req.token_ids,
            n_steps=req.n_steps or 100,
            cfg_weight=req.cfg_weight or 1.0,
            gt_pdb_string=req.gt_pdb_string,
        )
    elif req.tokenizer == TokenizerType.apt:
        result = bridge.decode_tokens(
            token_ids=req.token_ids,
            num_residues=req.num_residues,
            n_steps=req.n_steps or 100,
            gt_pdb_string=req.gt_pdb_string,
        )
    else:
        result = bridge.decode_tokens(
            token_ids=req.token_ids,
            atom_names=req.atom_names,
            residue_types=req.residue_types,
            residue_ids=req.residue_ids,
            chain_ids=req.chain_ids,
            gt_pdb_string=req.gt_pdb_string,
        )
    return DecodeResponse(pdb_string=result.pdb_string, num_atoms=result.num_atoms)


@app.post("/api/encode", response_model=EncodeResponse)
async def encode(
    file: UploadFile = File(...),
    tokenizer: TokenizerType = Query(TokenizerType.bio2token),
):
    bridge = _get_bridge(tokenizer)

    content = (await file.read()).decode("utf-8", errors="replace")

    # Convert CIF to PDB if needed
    if is_cif_format(content) or (file.filename and file.filename.lower().endswith(".cif")):
        content = cif_to_pdb(content)

    result = bridge.encode_pdb(content, filename=file.filename or "upload.pdb")
    return EncodeResponse(
        token_ids=result.token_ids,
        token_string=result.token_string,
        num_tokens=result.num_tokens,
        gt_pdb_string=result.gt_pdb_string,
        tokenizer=tokenizer.value,
        atom_names=result.atom_names,
        residue_types=result.residue_types,
        residue_ids=result.residue_ids,
        residue_names=result.residue_names,
        token_classes=result.token_classes,
        chain_ids=result.chain_ids,
        num_residues=result.num_residues,
        max_tokens=result.max_tokens,
    )


class PdbCodeRequest(BaseModel):
    pdb_code: str
    tokenizer: TokenizerType = TokenizerType.bio2token


@app.post("/api/encode-pdb-code", response_model=EncodeResponse)
async def encode_pdb_code(req: PdbCodeRequest):
    bridge = _get_bridge(req.tokenizer)

    code = req.pdb_code.strip().upper()
    if not re.match(r"^[A-Z0-9]{4}$", code):
        raise HTTPException(status_code=400, detail=f"Invalid PDB code: {req.pdb_code}")

    # Try PDB format first, fall back to CIF
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"https://files.rcsb.org/download/{code}.pdb")
        if resp.status_code == 200:
            content = resp.text
        else:
            resp = await client.get(f"https://files.rcsb.org/download/{code}.cif")
            if resp.status_code != 200:
                raise HTTPException(status_code=404, detail=f"PDB code {code} not found on RCSB")
            content = cif_to_pdb(resp.text)

    result = bridge.encode_pdb(content, filename=f"{code}.pdb")
    return EncodeResponse(
        token_ids=result.token_ids,
        token_string=result.token_string,
        num_tokens=result.num_tokens,
        gt_pdb_string=result.gt_pdb_string,
        tokenizer=req.tokenizer.value,
        atom_names=result.atom_names,
        residue_types=result.residue_types,
        residue_ids=result.residue_ids,
        residue_names=result.residue_names,
        token_classes=result.token_classes,
        chain_ids=result.chain_ids,
        num_residues=result.num_residues,
        max_tokens=result.max_tokens,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        bio2token_loaded=bio2token_bridge.model_loaded,
        apt_loaded=apt_available and apt_bridge is not None and apt_bridge.model_loaded,
        kanzi_loaded=kanzi_available and kanzi_bridge is not None and kanzi_bridge.model_loaded,
    )


# --- Static files ---

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    # Look for SSL certs next to pyproject.toml
    project_root = Path(__file__).resolve().parent.parent.parent
    cert_file = project_root / "cert.pem"
    key_file = project_root / "key.pem"

    ssl_kwargs = {}
    if cert_file.is_file() and key_file.is_file():
        ssl_kwargs["ssl_certfile"] = str(cert_file)
        ssl_kwargs["ssl_keyfile"] = str(key_file)
        print(f"\n  viz-bio2token starting at https://{host}:{port}\n")
    else:
        print(f"\n  viz-bio2token starting at http://{host}:{port}\n")

    uvicorn.run(app, host=host, port=port, **ssl_kwargs)


if __name__ == "__main__":
    main()
