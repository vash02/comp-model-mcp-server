# embedding_utils.py
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import hashlib, json

# 1) Text embedder for specs
TEXT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# 2) Numeric autoencoder (you can reuse the PyTorch AE from above)
class NumericAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)

# load or init a numeric autoencoder for each model_id
NUMERIC_AE_STORE = {}

def get_numeric_ae(dim, model_id=None):
    # for simplicity, share one model per model_id
    key = model_id or '__global__'
    if key not in NUMERIC_AE_STORE:
        ae = NumericAE(dim, latent_dim=min(8, dim//2)).eval()
        NUMERIC_AE_STORE[key] = ae
    return NUMERIC_AE_STORE[key]

def embed_model_spec(spec: dict) -> list:
    """
    spec: dict with keys 'equations', 'parameters', 'initial_conditions', 'param_ranges'
    We convert spec to a single text string and embed it.
    """
    eqs = "\n".join(spec['equations'])
    params = ", ".join(f"{p['name']}:{p['description']}" for p in spec['parameters'])
    ics   = ", ".join(f"{ic['name']}:{ic['description']}" for ic in spec.get('initial_conditions',[]))
    full = f"Eqs:\n{eqs}\nParams:\n{params}\nICs:\n{ics}"
    emb = TEXT_MODEL.encode(full, normalize_embeddings=True)
    return emb.tolist()

def embed_experiment(
    params: dict,
    ics: dict,
    results: dict,
    model_id: str,
    param_ranges: dict
) -> list:
    """
    1) Text part: small text summary of params+ics+results
    2) Numeric part: normalized vector [params|ics|results] -> latent via autoencoder
    3) Concat and return as list
    """
    # a) Text summary
    text = (
      "Params: " + ", ".join(f"{k}={params[k]}" for k in sorted(params)) +
      " | ICs: "   + ", ".join(f"{k}={ics[k]}"    for k in sorted(ics)) +
      " | Res: "  + ", ".join(f"{k}={results[k]}" for k in sorted(results))
    )
    txt_emb = TEXT_MODEL.encode(text, normalize_embeddings=True)

    # b) Numeric vector
    # build raw vector in a consistent order
    keys = sorted(params.keys()) + sorted(ics.keys()) + sorted(results.keys())
    vec = []
    for k in keys:
        val = None
        if k in params:
            val = float(params[k])
            lo, hi = param_ranges.get(k, (val, val))
        elif k in ics:
            val = float(ics[k])
            lo, hi = param_ranges.get(k, (val, val))
        else:
            val = float(results[k])
            # for results, you might not have ranges—scale by max observed? for now use raw
            lo, hi = (0.0, val) if val!=0 else (0.0, 1.0)
        # normalize
        norm = (val - lo)/(hi - lo) if hi>lo else 0.0
        vec.append(norm)
    vec = np.array(vec, dtype=np.float32)

    # pass through numeric AE
    ae = get_numeric_ae(len(vec), model_id)
    with torch.no_grad():
        num_emb = ae.encoder(torch.from_numpy(vec).unsqueeze(0)).squeeze(0).numpy()

    # c) Concatenate
    joint = np.concatenate([np.array(txt_emb, dtype=np.float32), num_emb], axis=0)
    return joint.tolist()
