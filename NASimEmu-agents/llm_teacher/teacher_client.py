"""
The only file that ever calls the LLM API -- master plan Sec 15.13.

Teacher = a locally-hosted, free, open-weight model served by Ollama
(http://localhost:11434), not a paid hosted API. Chosen deliberately: Sec
15.2 already requires the teacher to run strictly offline (never inside
forward()), so nothing in this design needs low latency or a live network
call during training -- a free local model removes per-call cost entirely
for the labeling pass, at the cost of needing it installed on the labeling
machine.

Uses Ollama's /api/chat (not /api/generate): with /api/generate, a
hybrid-thinking model's JSON answer could land in the response's `thinking`
channel instead of `response` and look like an empty answer unless you went
digging (observed with qwen3:4b during this project's laptop-scale model
comparison, which is why qwen2.5:3b-instruct -- no thinking channel -- became
the known-working baseline). /api/chat's `message.content` is unambiguously
the final answer; `message.thinking` (if the model has one) is recorded
separately for audit, never silently substituted in if `content` is empty --
an empty `content` is a real, comparable failure data point for that model,
not something to paper over. See docs/llm_teacher_workstation_upgrade_walkthrough.md
for the full model-comparison methodology this migration supports (workstation-only
candidates like mistral-small3.2:24b / qwen3.5:9b -- DEFAULT_MODEL here stays
qwen2.5:3b-instruct, the only one actually validated on this task so far).

This module never trains anything and is never imported by nasim_net_base_hrl.py
or main.py's core training loop -- only by label_states.py and cascade.py,
which run as a separate, offline, one-time (or DAgger-style re-run)
data-collection pass.
"""
import hashlib
import json
import time
import urllib.request
import urllib.error

from .output_schema import JSON_SCHEMA, OUTPUT_SCHEMA_VERSION
from .prompt_templates import SYSTEM_PROMPT, build_user_prompt, PROMPT_VERSION
from .validator import parse_and_validate

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_HOST = "http://localhost:11434"
MAX_RETRIES = 3
DEFAULT_NUM_CTX = 8192  # NASimEmu's state summaries are short; don't over-allocate KV cache


def summary_hash(summary):
    blob = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _call_ollama_chat(prompt, model, host, timeout_s, system_prompt=SYSTEM_PROMPT, schema=JSON_SCHEMA):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "format": schema,
        "stream": False,
        "think": False,  # only affects hybrid-thinking models; harmless no-op otherwise
        "options": {"temperature": 0, "num_ctx": DEFAULT_NUM_CTX},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    message = body.get("message", {})
    content = message.get("content", "") or body.get("response", "")  # fallback for one release cycle
    thinking = message.get("thinking")
    return content, thinking


def _label_with_prompt(prompt, summary, model, host, timeout_s, system_prompt=SYSTEM_PROMPT):
    """Shared retry-loop core: query the teacher with an arbitrary prompt
    against a given state summary (used for both the standard goal-selection
    prompt and cascade.py's critique/escalation prompt, which asks a
    different question over the same summary), validate, and return a full
    provenance record. Retries up to MAX_RETRIES times on invalid/unparsable
    output before giving up and recording the last attempt as rejected
    (never silently discarded, Sec 15.7)."""
    last_raw, last_parsed, last_valid, last_reason, last_thinking = None, None, False, "no_attempt", None
    latency_s = 0.0
    for _attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            raw_text, thinking = _call_ollama_chat(prompt, model, host, timeout_s, system_prompt=system_prompt)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_raw, last_parsed, last_valid, last_reason, last_thinking = None, None, False, f"api_error:{e}", None
            break
        latency_s = time.time() - t0

        parsed, is_valid, reject_reason = parse_and_validate(raw_text, summary)
        last_raw, last_parsed, last_valid, last_reason, last_thinking = raw_text, parsed, is_valid, reject_reason, thinking
        if is_valid:
            break

    return {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "input_hash": summary_hash(summary),
        "state_summary": summary,
        "raw_output": last_raw,
        "thinking": last_thinking,  # audit/debug only -- never substituted for raw_output if empty
        "parsed_output": last_parsed,
        "valid": bool(last_valid),
        "reject_reason": last_reason,
        "latency_s": latency_s,
        "generated_at": time.time(),
    }


def label_one_state(summary, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout_s=120):
    """
    Queries the local teacher for a single sanitized state summary, validates
    the result, and returns a full provenance record (Sec 15.14): model,
    versions, input hash, raw + parsed output, validity, and latency.
    """
    prompt = build_user_prompt(summary)
    return _label_with_prompt(prompt, summary, model, host, timeout_s)
