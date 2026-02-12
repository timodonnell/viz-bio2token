# viz-bio2token

Web-based visualization tool for [bio2token](https://github.com/flagshippioneering/bio2token) ([paper](https://arxiv.org/pdf/2410.19110)) protein structure encodings.

Split-pane UI: tokenized bio2token input on the left, live 3D structure rendering on the right.

![screenshot](https://img.shields.io/badge/3Dmol.js-viewer-blue)

## Features

- **Load by PDB code** — enter a 4-letter PDB code (e.g. `6O4Y`) to fetch from RCSB, tokenize, and visualize
- **Upload PDB/CIF** — drag-and-drop or upload a local structure file
- **Live editing** — edit tokens in the left pane and see the decoded structure update in real time (500ms debounce)
- **Original vs. decoded overlay** — toggle "Show original" to compare the ground-truth structure (gray) against the decoded reconstruction, Kabsch-aligned for proper comparison
- **Per-chain coloring** — each protein chain is rendered in a distinct color
- **Atom-token linking** — click an atom in the 3D viewer to highlight the corresponding token in the editor
- **Rendering styles** — cartoon, stick, sphere, or line
- **Metadata panel** — view/edit atom names, residue types, and residue IDs used for decoding

## Quickstart

Requires CUDA (mamba-ssm dependency) and a bio2token checkpoint.

```bash
# Clone
git clone https://github.com/timodonnell/viz-bio2token.git
cd viz-bio2token

# Run (installs dependencies automatically via uv)
uv run viz-bio2token
```

The server starts at `http://0.0.0.0:8000`. It loads the bio2token model on startup, then auto-loads PDB `6O4Y` as a demo.

### HTTPS

To run with HTTPS (avoids browser warnings on remote machines), place `cert.pem` and `key.pem` in the project root:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'
uv run viz-bio2token
```

The server will automatically detect the certs and serve over HTTPS.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIO2TOKEN_CHECKPOINT` | — | Path to a `.ckpt` file. If not set, searches `/home/ubuntu/bio2token/checkpoints/` then `./checkpoints/` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Architecture

```
Browser                          Server (FastAPI + uvicorn)
┌──────────────────────┐         ┌──────────────────────────┐
│ Left pane            │         │                          │
│  Token textarea      │◄───────►│  POST /api/decode        │
│  Metadata panel      │  JSON   │  POST /api/encode        │
│                      │         │  POST /api/encode-pdb-code│
│ Right pane           │         │  GET  /api/status        │
│  3Dmol.js viewer     │         │                          │
└──────────────────────┘         │  bio2token               │
                                 │   Mamba encoder (4 layers)│
                                 │   FSQ quantizer (4096)   │
                                 │   Mamba decoder (6 layers)│
                                 └──────────────────────────┘
```

## File Structure

```
pyproject.toml                          # uv project config, dependencies, entry point
src/viz_bio2token/
  app.py                                # FastAPI app, API endpoints, main()
  bio2token_bridge.py                   # Wraps bio2token Encoder/Decoder/FSQ
  pdb_utils.py                          # CIF→PDB conversion (BioPython)
static/
  index.html                            # Split-pane layout, 3Dmol.js CDN
  style.css                             # Dark theme, flexbox layout
  app.js                                # Token parsing, API calls, 3Dmol rendering
```

## How It Works

1. **Encoding**: A PDB/CIF file is parsed via bio2token's `pdb_2_dict` + `uniform_dataframe` pipeline, then passed through the Mamba encoder and FSQ quantizer to produce integer token IDs (0–4095). Metadata (atom names, residue types, residue IDs, chain IDs) is returned alongside tokens.

2. **Decoding**: Token IDs are converted back to 128-dim vectors via `FSQ.indices_to_codes()`, then passed through the Mamba decoder to produce 3D coordinates. When a ground-truth structure is available, the decoded coordinates are Kabsch-aligned (rigid body) for visual comparison.

3. **Rendering**: The frontend uses [3Dmol.js](https://3dmol.csb.pitt.edu/) to display PDB structures. Each chain gets a distinct color. Clicking an atom highlights the corresponding token in the editor.
