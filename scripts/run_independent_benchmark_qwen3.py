"""Re-run the 27-scenario independent adversarial benchmark with a second model family (qwen3:8b).

The only change from the llama3 run is the model and two Ollama request settings that qwen3 needs:
thinking disabled ("think": False), so the answer is a plain sentence like llama3's, and a longer
timeout for the first, model-loading call. Everything else is the project's own runner, unchanged.

    python scripts/run_independent_benchmark_qwen3.py
    -> results/independent_adversarial_benchmark_qwen3_8b.json
"""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisos.agent import model_backend
from aisos.agent.model_backend import GenerationResult, OllamaBackend


def _call(self, prompt: str, system: str = "") -> GenerationResult:
    payload = {
        "model": self.model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": {"temperature": self.temperature},
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
    latency_s = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return GenerationResult(
        text=data.get("response", "").strip(),
        input_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
        latency_s=latency_s,
        model_id=self.model_id,
    )


OllamaBackend._call = _call
model_backend.OllamaBackend._call = _call

from aisos.benchmark.independent_adversarial import run_independent_benchmark as rib

rib.run(model_id="qwen3:8b")
