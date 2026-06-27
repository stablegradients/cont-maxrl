"""Add `user_prompt` + `prompt_rendered` to each problem record in the prerl
gen_*.json dumps and rewrite them pretty-printed (indent=2) for text-editor
viewing. Reconstructs the exact prompt the worker fed the model from
problems32.json (src_code) + the config's prompt template + the model's chat
template. Idempotent; safe to re-run.

  python -m experiment11_code_optim.eval.prerl_addprompt
"""
import glob, json, os, sys
from pathlib import Path

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
os.environ.setdefault("HF_HOME", "/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface")
from experiment11_code_optim.eval.prerl_gen_score import PROMPTS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/results/prerl_analysis"
src_by_key = {(p["problem_id"], p["src_id"]): p["src_code"]
              for p in json.load(open(f"{OUT}/problems32.json"))}

_tok: dict = {}
def _t(m):
    if m not in _tok:
        _tok[m] = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    return _tok[m]

def render(model, body, think_mode):
    msgs = [{"role": "user", "content": body}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    if think_mode == "on":
        kw["enable_thinking"] = True
    elif think_mode == "off":
        try:
            return _t(model).apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            pass
    return _t(model).apply_chat_template(msgs, **kw)

FRONT = ["problem_id", "src_id", "ceiling_speedup", "our_speedup_usable", "our_fastest_speedup",
         "n_usable", "config_name", "model", "prompt", "think_mode", "user_prompt", "prompt_rendered"]

for f in sorted(glob.glob(f"{OUT}/gen_*.json")):
    if f.endswith("_smoke.json"):
        continue
    d = json.load(open(f))
    for i, rec in enumerate(d):
        src = src_by_key.get((rec["problem_id"], rec["src_id"]))
        body = PROMPTS[rec["prompt"]].format(src=src) if src is not None else None
        rec["user_prompt"] = body
        rec["prompt_rendered"] = render(rec["model"], body, rec["think_mode"]) if body is not None else None
        d[i] = {**{k: rec[k] for k in FRONT if k in rec},
                **{k: v for k, v in rec.items() if k not in FRONT}}
    Path(f).write_text(json.dumps(d, indent=2))
    print(f"updated {Path(f).name}  ({Path(f).stat().st_size/1e6:.1f} MB, {len(d)} problems)")
print("done")
