"""FastAPI application for viz-bio2token."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from viz_bio2token.bio2token_bridge import Bio2TokenBridge
from viz_bio2token.pdb_utils import cif_to_pdb, is_cif_format

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

bridge = Bio2TokenBridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the bio2token model on startup."""
    print("Loading bio2token model...")
    bridge.load()
    yield


app = FastAPI(title="viz-bio2token", lifespan=lifespan)


# --- Request/Response models ---

class DecodeRequest(BaseModel):
    token_ids: list[int]
    atom_names: Optional[list[str]] = None
    residue_types: Optional[list[str]] = None
    residue_ids: Optional[list[int]] = None
    gt_pdb_string: Optional[str] = None


class DecodeResponse(BaseModel):
    pdb_string: str
    num_atoms: int


class EncodeResponse(BaseModel):
    token_ids: list[int]
    token_string: str
    num_tokens: int
    gt_pdb_string: str
    atom_names: list[str]
    residue_types: list[str]
    residue_ids: list[int]
    residue_names: list[str]
    token_classes: list[int]


class StatusResponse(BaseModel):
    model_loaded: bool


# --- API endpoints ---

@app.post("/api/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest):
    if not bridge.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate token range
    for tid in req.token_ids:
        if tid < 0 or tid > 4095:
            raise HTTPException(status_code=400, detail=f"Token ID {tid} out of range [0, 4095]")

    if len(req.token_ids) == 0:
        raise HTTPException(status_code=400, detail="No token IDs provided")

    result = bridge.decode_tokens(
        token_ids=req.token_ids,
        atom_names=req.atom_names,
        residue_types=req.residue_types,
        residue_ids=req.residue_ids,
        gt_pdb_string=req.gt_pdb_string,
    )
    return DecodeResponse(pdb_string=result.pdb_string, num_atoms=result.num_atoms)


@app.post("/api/encode", response_model=EncodeResponse)
async def encode(file: UploadFile = File(...)):
    if not bridge.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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
        atom_names=result.atom_names,
        residue_types=result.residue_types,
        residue_ids=result.residue_ids,
        residue_names=result.residue_names,
        token_classes=result.token_classes,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(model_loaded=bridge.model_loaded)


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
