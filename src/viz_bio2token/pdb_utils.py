"""Utilities for CIF/PDB format detection and conversion."""

from __future__ import annotations

import io
import tempfile
import os


def is_cif_format(content: str) -> bool:
    """Detect whether file content is in mmCIF format."""
    for line in content.splitlines()[:50]:
        stripped = line.strip()
        if stripped.startswith("data_") or stripped.startswith("loop_") or stripped.startswith("_"):
            return True
    return False


def cif_to_pdb(content: str) -> str:
    """Convert mmCIF content to PDB format string via BioPython."""
    from Bio.PDB import MMCIFParser, PDBIO

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.write(content)
        cif_path = f.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_path)

        pdb_io = PDBIO()
        pdb_io.set_structure(structure)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as out:
            pdb_path = out.name

        pdb_io.save(pdb_path)

        with open(pdb_path, "r") as f:
            pdb_content = f.read()

        return pdb_content
    finally:
        os.unlink(cif_path)
        if "pdb_path" in dir():
            try:
                os.unlink(pdb_path)
            except OSError:
                pass
