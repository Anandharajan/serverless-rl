"""Lambda handler for the sim-to-real calibration: real CPU inference of Qwen2.5-0.5B (GGUF).

The model artefact is downloaded from S3 into /tmp during INIT, so a cold start includes the
real model-loading path (S3 transfer + llama.cpp load). Each response reports server-side
phase timings; CloudWatch REPORT lines give INIT and billed durations.
"""
import json
import os
import time

T_IMPORT0 = time.time()
import boto3  # noqa: E402
from llama_cpp import Llama  # noqa: E402

MODEL_KEY = os.environ.get("MODEL_KEY", "qwen2.5-0.5b-instruct-q8_0.gguf")
BUCKET = os.environ.get("MODEL_BUCKET", "")
LOCAL = os.environ.get("MODEL_LOCAL", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))


def _threads():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 1


INIT = {"import_s": time.time() - T_IMPORT0}
_t = time.time()
if LOCAL:
    path = LOCAL
else:
    path = "/tmp/" + MODEL_KEY
    if not os.path.exists(path):
        boto3.client("s3").download_file(BUCKET, MODEL_KEY, path)
INIT["download_s"] = time.time() - _t
_t = time.time()
LLM = Llama(model_path=path, n_ctx=2048, n_threads=_threads(), n_batch=512, verbose=False)
INIT["load_s"] = time.time() - _t
INIT["threads"] = _threads()
INIT["model_mb"] = os.path.getsize(path) / 1e6
_first = {"cold": True}

FILLER = ("Serverless platforms allocate CPU in proportion to configured memory, and an inference "
          "function must load its model before serving the first request. ")


_N = {"i": 0}


def make_prompt(kind: str) -> str:
    reps = 4 if kind == "short" else 22  # ~90 vs ~500 input tokens
    _N["i"] += 1
    # unique prefix defeats prompt-prefix KV reuse, so every request pays full prefill
    return f"[request {time.time_ns()}-{_N['i']}] Summarise the following text in one sentence.\n\n" + FILLER * reps


def handler(event, context):
    kind = event.get("kind", "short")
    LLM.reset()
    t0 = time.time()
    out = LLM(make_prompt(kind), max_tokens=int(event.get("max_tokens", MAX_TOKENS)),
              temperature=0.0)
    dt = time.time() - t0
    cold = _first["cold"]
    _first["cold"] = False
    u = out.get("usage", {})
    return {"kind": kind, "service_s": dt, "cold": cold, "init": INIT if cold else None,
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "model": MODEL_KEY,
            "mem_mb": int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "0"))}


if __name__ == "__main__":  # local benchmark
    import sys
    for kind in ("short", "long"):
        r = [handler({"kind": kind}, None)["service_s"] for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)]
        print(json.dumps({"kind": kind, "threads": INIT["threads"], "model": MODEL_KEY,
                          "load_s": round(INIT["load_s"], 3), "median_s": sorted(r)[len(r) // 2],
                          "all": [round(x, 3) for x in r]}))
