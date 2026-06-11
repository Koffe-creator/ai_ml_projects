"""Patch the installed `geneformer` package to run on CPU.

The published package hardcodes `device="cuda"` in a couple of spots used by
EmbExtractor.extract_embs, which crashes on machines without a CUDA GPU
(e.g. macOS). This script rewrites those lines in-place to fall back to CPU.

Run once after installing geneformer:
    python scripts/patch_geneformer_cpu.py
"""

import re
from pathlib import Path

import geneformer

PKG_DIR = Path(geneformer.__file__).parent

PATCHES = {
    PKG_DIR / "emb_extractor.py": [
        (r'device="cuda"\)', 'device=model.device)'),
        (r'\.to\("cuda"\),', ".to(model.device),"),
    ],
    PKG_DIR / "perturber_utils.py": [
        (r'device="cuda"', 'device="cpu" if not torch.cuda.is_available() else "cuda"'),
        (r'\.to\("cuda"\)', '.to("cpu" if not torch.cuda.is_available() else "cuda")'),
    ],
}


def main():
    for path, patterns in PATCHES.items():
        text = path.read_text()
        original = text
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        if text != original:
            path.write_text(text)
            print(f"Patched {path}")
        else:
            print(f"No changes needed (already patched?): {path}")


if __name__ == "__main__":
    main()
