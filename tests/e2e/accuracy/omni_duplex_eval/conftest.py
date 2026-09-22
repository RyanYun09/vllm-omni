# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Module-scoped judge server fixture for omni-duplex-eval CI guard."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Generator

import pytest
import requests
import torch


@pytest.fixture(scope="module")
def judge_server() -> Generator[str, None, None]:
    """Launch the Qwen2.5-VL-7B judge server, wait for health, yield base URL.

    If ``VLLM_DUPLEX_EVAL_JUDGE_URL`` is already set *and* the endpoint
    responds to ``/health``, reuse it (developer convenience).

    Otherwise, start the judge as a subprocess:
    - On CUDA hosts: pin the judge to ``CUDA_VISIBLE_DEVICES=0`` so the
      omni server (managed by the ``omni_server`` fixture) gets GPU 1.
    - On NPU hosts: pin to ``ASCEND_RT_VISIBLE_DEVICES=0``; the omni
      server uses device 1 via ``ASCEND_RT_VISIBLE_DEVICES=1`` in the YAML.

    The fixture terminates the judge on teardown.
    """
    port = int(os.environ.get("VLLM_DUPLEX_EVAL_JUDGE_PORT", "8001"))
    base_url = f"http://127.0.0.1:{port}"

    # Reuse a running judge if URL is pre-set and healthy.
    preset_url = os.environ.get("VLLM_DUPLEX_EVAL_JUDGE_URL")
    if preset_url:
        try:
            resp = requests.get(f"{preset_url}/health", timeout=5)
            if resp.status_code == 200:
                yield preset_url
                return
        except Exception:
            pass

    # Build device-isolated environment.
    env = os.environ.copy()
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = "0"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        env["ASCEND_RT_VISIBLE_DEVICES"] = "0"

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "--port",
        str(port),
        "--max-model-len",
        "65536",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--allowed-local-media-path",
        "/tmp/omni_duplex_ci",
    ]

    proc = subprocess.Popen(cmd, env=env)

    # Health-check loop: up to 60 iterations × 5 s = 5 min.
    for _ in range(60):
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Judge server failed to become healthy at {base_url} within 300 s")

    yield base_url

    # Teardown.
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
