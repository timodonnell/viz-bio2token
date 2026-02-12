# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

viz-bio2token is a web-based visualization tool for [bio2token](https://github.com/flagshippioneering/bio2token) ([paper](https://arxiv.org/pdf/2410.19110)) encodings. It displays a split-pane UI: tokenized bio2token input on the left, live 3D structure rendering on the right.

## Running

```bash
uv run viz-bio2token
```

This starts a FastAPI server on `http://0.0.0.0:8000`. Requires CUDA (mamba-ssm dependency).

### Environment Variables

- `BIO2TOKEN_CHECKPOINT` — path to a `.ckpt` file. If not set, looks in `/home/ubuntu/bio2token/checkpoints/` then `./checkpoints/`.
- `HOST` — server bind address (default `0.0.0.0`)
- `PORT` — server port (default `8000`)

## Architecture

- **Package manager**: uv
- **Backend**: FastAPI + uvicorn, serves API and static files
- **Frontend**: Vanilla HTML/CSS/JS with 3Dmol.js (CDN)
- **Key dependencies**: bio2token (local editable), torch 2.4.1+cu121, mamba-ssm 2.2.2, transformers <5

## File Structure

```
pyproject.toml                          # uv project config, dependencies, entry point
src/viz_bio2token/
  __init__.py
  app.py                                # FastAPI app, endpoints, lifespan, main()
  bio2token_bridge.py                   # Wraps bio2token Encoder/Decoder/FSQ for encode & decode
  pdb_utils.py                          # CIF detection and CIF→PDB conversion (BioPython)
static/
  index.html                            # Split-pane layout, 3Dmol.js CDN
  style.css                             # Dark theme, flexbox layout, draggable divider
  app.js                                # Token parsing, debounced API calls, 3Dmol rendering, metadata UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves index.html |
| `/api/decode` | POST | Decode token IDs → PDB string. Accepts optional metadata (atom_names, residue_types, residue_ids) |
| `/api/encode` | POST | Upload PDB/CIF file → token IDs + metadata + ground-truth PDB |
| `/api/status` | GET | Check if model is loaded |

## Key Design Decisions

- Encoder and Decoder are instantiated separately (not the full Autoencoder) to avoid needing Registration/Losses modules during inference
- FSQ `indices_to_codes()` converts integer token IDs → 128-dim vectors for the decoder
- When encoding a PDB, metadata (atom_names, residue_types, residue_ids) is returned alongside tokens and stored in the frontend, then sent back with decode requests for proper PDB labeling
- Without metadata, decode falls back to CA/ALA placeholders
- The metadata panel in the UI allows users to view and edit atom names, residue types, and residue IDs
