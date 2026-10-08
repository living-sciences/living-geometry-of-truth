"""Smoke test the 2026 MM-arch model (Qwen3.5-9B-Base) with plain transformers
(NOT nnsight): load, one forward, read a residual-stream activation via output_hidden_states.
"""
import sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"

torch.set_grad_enabled(False)
print("loading", PATH, flush=True)
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained(PATH)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("model_type:", m.config.model_type)
print("top-level children:", [n for n, _ in m.named_children()])
if hasattr(m, "model"):
    print("m.model children:", [n for n, _ in m.model.named_children()])

# locate the text decoder (module holding .layers = residual stream)
def find_decoder(root):
    for name, mod in root.named_modules():
        if name.endswith("layers") and hasattr(mod, "__len__") and len(mod) == 32:
            parent = root
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            return parent, name
    return None, None

dec, decpath = find_decoder(m)
print("decoder path (.layers at):", decpath)
print("n decoder layers:", len(dec.layers), "hidden:", m.config.text_config.hidden_size)

stmts = ["The city of Paris is in France.", "The city of Paris is in Japan."]
ids = tok(stmts, return_tensors="pt", padding=True).to(m.device)
print("input_ids shape:", tuple(ids.input_ids.shape), "padding_side:", tok.padding_side)

out = m(**ids, output_hidden_states=True)
hs = out.hidden_states
print("len(hidden_states):", len(hs), "(expect n_layers+1 = 33)")
L = 13  # round(0.4*32)
act = hs[L + 1][:, -1, :]  # output of decoder layer L, last token
print(f"layer {L} last-token act shape:", tuple(act.shape), act.dtype)
print("act[0][:5]:", act[0][:5].float().cpu().tolist())

# cross-check: forward hook on dec.layers[L] must match hidden_states[L+1]
cap = {}
def hook(mod, inp, o):
    cap['o'] = o[0] if isinstance(o, tuple) else o
h = dec.layers[L].register_forward_hook(hook)
_ = m(**ids)
h.remove()
hook_act = cap['o'][:, -1, :]
diff = (hook_act.float() - act.float()).abs().max().item()
print("max|hook - hidden_states[L+1]| last-token:", diff)
print("SMOKE OK")
