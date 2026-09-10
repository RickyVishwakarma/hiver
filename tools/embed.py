"""Stage B.2 - encode text to vectors for clustering and retrieval.

Uses transformers + torch directly with mean pooling instead of the
sentence-transformers package. Reasons (see DECISIONS.md):
  - one fewer dependency, and no version conflict with transformers 5.x
  - it is ~20 lines we can explain and modify live, which the brief warns about
  - mean pooling + L2 norm reproduces all-MiniLM-L6-v2's intended usage exactly

Vectors are cached to .tmp/embeddings/<name>.npy keyed by a hash of the input
texts, so re-running is free and the expensive step never repeats by accident.

Importable:  from embed import encode, load_or_encode
CLI:         python tools/embed.py --field opening_msg
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np

from common import EMBED_CACHE, EMBED_MODEL, THREADS, info, ok, read_jsonl, set_seed

_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _load_model():
    """Lazy global load - importing this module must stay cheap."""
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    import torch
    from transformers import AutoModel, AutoTokenizer

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"loading {EMBED_MODEL} on {_DEVICE}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL)
    _MODEL = AutoModel.from_pretrained(EMBED_MODEL).to(_DEVICE).eval()
    return _MODEL, _TOKENIZER, _DEVICE


def encode(texts: list[str], batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence embeddings.

    batch_size 64 @ 128 tokens keeps a MiniLM comfortably inside 4GB VRAM.
    """
    import torch

    model, tok, device = _load_model()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else "empty" for t in texts[i : i + batch_size]]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state  # (B, T, H)
        # Mean pool over real tokens only - padding must not dilute the vector.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy().astype(np.float32))
        if i and i % (batch_size * 50) == 0:
            info(f"  encoded {i:,}/{len(texts):,}")
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def _key(texts: list[str], tag: str) -> str:
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    h.update(str(len(texts)).encode())
    for t in texts[:2000]:  # sampling the head is enough to detect changes
        h.update(t.encode("utf-8", "ignore"))
    return f"{tag}-{h.hexdigest()[:16]}"


def load_or_encode(texts: list[str], tag: str = "text") -> np.ndarray:
    """Cached encode. Safe to call repeatedly across scripts."""
    path = EMBED_CACHE / f"{_key(texts, tag)}.npy"
    if path.exists():
        vecs = np.load(path)
        if vecs.shape[0] == len(texts):
            info(f"embeddings cache hit: {path.name} {vecs.shape}")
            return vecs
    vecs = encode(texts)
    np.save(path, vecs)
    ok(f"cached embeddings -> {path.name} {vecs.shape}")
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="opening_msg", help="Thread field to encode")
    ap.add_argument("--threads", default=str(THREADS))
    args = ap.parse_args()
    set_seed()

    rows = read_jsonl(args.threads)
    if not rows:
        raise SystemExit(f"no threads at {args.threads} - run build_threads.py first")
    texts = [r.get(args.field, "") or "" for r in rows]
    vecs = load_or_encode(texts, tag=args.field)
    ok(f"{vecs.shape[0]:,} vectors of dim {vecs.shape[1]}")
    # Quick self-check: a vector should be unit length and self-similarity 1.0
    info(f"sanity: ||v0||={np.linalg.norm(vecs[0]):.4f}  v0*v0={float(vecs[0] @ vecs[0]):.4f}")


if __name__ == "__main__":
    main()
