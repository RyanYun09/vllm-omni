# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Orchestrator dispatch wrap-once semantics (RFC #4872).

``invoke_orchestrator_processor`` runs per request (and per forward), so
``wrap_orchestrator_processor`` must not re-wrap — and must not re-emit the
legacy-contract ``DeprecationWarning`` — on every forward.  The wrap result is
memoized per callable: the signature probe and the warning happen **once** per
processor (at first resolution/invocation), then the cached C1-compatible
wrapper is reused.

Pure-logic, CPU-only (no model loading, no vllm runtime).
"""

from __future__ import annotations

from typing import Any

import pytest

from vllm_omni.model_executor.stage_input_processors._dispatch import (
    OrchestratorInputContext,
    invoke_orchestrator_processor,
    wrap_orchestrator_processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _legacy_c2(source_outputs, prompt=None, requires_multimodal_data=False, streaming_context=None):
    """A legacy C2 positional shell (like qwen3_omni.thinker2talker_token_only)."""
    return [source_outputs, prompt, requires_multimodal_data, streaming_context]


def _legacy_c0(source_outputs, prompt=None, requires_multimodal_data=False):
    """A legacy C0 3-arg shell."""
    return [source_outputs, prompt, requires_multimodal_data]


def _c1(source_outputs: list[Any], ctx: OrchestratorInputContext):
    """Already speaks the C1 ``(source_outputs, ctx)`` contract."""
    return [source_outputs, ctx]


def _count_deprecation_warnings(trigger) -> int:
    """Run *trigger* under ``simplefilter("always")`` and count DeprecationWarnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger()
    return sum(1 for w in caught if issubclass(w.category, DeprecationWarning))


def test_wrap_legacy_shape_warns_once_then_memoized():
    first = _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2))
    # The first wrap (at "resolve") emits exactly one DeprecationWarning.
    assert first == 1
    # Subsequent wraps reuse the memoized adapter: no further warning.
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2)) == 0
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2)) == 0


def test_invoke_does_not_rewarn_on_every_forward():
    ctx = OrchestratorInputContext(prompt="p", requires_multimodal_data=True, streaming_context="s")

    def invoke_many():
        for _ in range(5):
            invoke_orchestrator_processor(_legacy_c0, ["out"], ctx)

    # Exactly one DeprecationWarning total across 5 forwards (wrap happens once).
    assert _count_deprecation_warnings(invoke_many) == 1


def test_invoke_adapts_legacy_shape_and_forwards_context():
    ctx = OrchestratorInputContext(prompt="p", requires_multimodal_data=True, streaming_context="s")
    result = invoke_orchestrator_processor(_legacy_c2, ["out"], ctx)
    # C2 adapter forwards (source_outputs, prompt, requires_multimodal_data, streaming_context).
    assert result == [["out"], "p", True, "s"]


def test_c1_processor_passes_through_without_warning():
    wrapped = wrap_orchestrator_processor(_c1)
    assert wrapped is _c1  # already speaks ctx: returned unchanged
    ctx = OrchestratorInputContext(prompt="p")
    assert invoke_orchestrator_processor(_c1, ["out"], ctx) == [["out"], ctx]
    # No DeprecationWarning for C1 processors at all.
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_c1)) == 0
