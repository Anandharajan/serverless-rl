"""Load driver for the sim-to-real calibration (runs as its own Lambda, boto3 only).

event = {
  "target": "function name or ARN (optionally :alias)",
  "schedule": [{"t": seconds_from_start, "kind": "short"|"long", "target": optional override}],
  "max_workers": 40,
  "out_bucket": "...", "out_key": "results/<exp>.jsonl",
  "tag": "free-form experiment label"
}
Each request is sent at its scheduled offset (open-loop), synchronously, with LogType=Tail so
the REPORT line (Duration, Billed Duration, Init Duration) is captured per request.
"""
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

LAM = boto3.client("lambda", config=Config(read_timeout=900, max_pool_connections=64,
                                           retries={"max_attempts": 0}))
S3 = boto3.client("s3")
REPORT = re.compile(r"REPORT RequestId: (\S+)\s+Duration: ([\d.]+) ms\s+Billed Duration: (\d+) ms"
                    r"\s+Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?")


def _call(target, kind, t_sched, t0, tag):
    delay = t0 + t_sched - time.time()
    if delay > 0:
        time.sleep(delay)
    sent = time.time()
    rec = {"tag": tag, "target": target, "kind": kind, "t_sched": t_sched, "sent": sent - t0}
    try:
        r = LAM.invoke(FunctionName=target, InvocationType="RequestResponse", LogType="Tail",
                       Payload=json.dumps({"kind": kind}).encode())
        rec["latency_s"] = time.time() - sent
        rec["status"] = r.get("StatusCode")
        rec["fn_error"] = r.get("FunctionError")
        body = r["Payload"].read()
        try:
            rec["resp"] = json.loads(body)
        except Exception:
            rec["resp"] = body[:300].decode(errors="replace")
        log = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
        m = REPORT.search(log)
        if m:
            rec.update({"duration_ms": float(m.group(2)), "billed_ms": int(m.group(3)),
                        "mem_mb": int(m.group(4)), "max_mem_used_mb": int(m.group(5)),
                        "init_ms": float(m.group(6)) if m.group(6) else None})
    except LAM.exceptions.TooManyRequestsException:
        rec.update({"latency_s": time.time() - sent, "throttled": True})
    except Exception as e:  # noqa: BLE001
        rec.update({"latency_s": time.time() - sent, "error": repr(e)[:300]})
    return rec


def handler(event, context):
    sched = sorted(event["schedule"], key=lambda x: x["t"])
    t0 = time.time() + 1.0
    out = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=int(event.get("max_workers", 40))) as ex:
        futs = [ex.submit(_call, s.get("target", event["target"]), s["kind"], s["t"], t0,
                          event.get("tag", "")) for s in sched]
        for f in futs:
            r = f.result()
            with lock:
                out.append(r)
    body = "\n".join(json.dumps(r) for r in out)
    S3.put_object(Bucket=event["out_bucket"], Key=event["out_key"], Body=body.encode())
    lat = sorted(r.get("latency_s", 0) for r in out)
    return {"n": len(out), "p50": lat[len(lat) // 2] if lat else None,
            "p99": lat[int(0.99 * (len(lat) - 1))] if lat else None,
            "errors": sum(1 for r in out if r.get("error") or r.get("fn_error") or r.get("throttled")),
            "key": event["out_key"]}
