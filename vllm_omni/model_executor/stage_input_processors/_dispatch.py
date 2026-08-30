# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""RFC #4872 Phase 2 (P1): orchestrator input-dispatch contract layer.

This module defines the **consumer-side (orchestrator / stage-client) dispatch
contract** for ``stage_input_processors``.  It replaces the two inconsistent
``inspect.signature`` probes in the orchestrator runtime with one shared
contract:

* ``stage_engine_core_client.process_engine_inputs`` used an **arity** probe
  (``len(signature.parameters) >= 4``) to pick a 3-arg vs 4-arg positional
  call.
* ``orchestrator._forward_to_next_stage`` (diffusion branch) used a **name**
  probe (``"sampling_params" in signature.parameters``) to decide whether to
  inject the diffusion-stage sampling params as a keyword.

Both are collapsed into a single normalization layer:

* ``OrchestratorInputContext`` — the fixed transition context the runtime
  passes to every consumer-side builder.
* ``PlaceholderPromptBuilder`` / ``DiffusionInputBuilder`` — the two
  orchestrator-facing callable roles (C1-compatible ``(source_outputs, ctx)``).
* ``wrap_orchestrator_processor`` / ``invoke_orchestrator_processor`` —
  processors that already speak the ``ctx`` contract are passed through
  unchanged; legacy positional shapes (C0/C2/C3/C4 below) are adapted with a
  ``DeprecationWarning``.

**Producer roles are intentionally out of scope here.**  ``FullPayloadProducer``
(``*, transfer_manager, pooling_output, request, is_finished=...``) and
``AsyncChunkProducer`` (``*, transfer_manager, multimodal_output, request,
is_finished=False``) run inside workers and never receive an
``OrchestratorInputContext``; their ``pooling_output`` / ``multimodal_output``
keyword names are load-bearing parts of the producer contract and are kept
as-is (RFC P1/P2 keep the producer kwargs contract unchanged).

Legacy positional shapes normalized here (RFC P1 migration matrix):

* C0 3-arg: ``(source_outputs, prompt, requires_multimodal_data)``.
* C2 placeholder: ``(source_outputs, prompt, requires_multimodal_data,
  streaming_context=None)`` (the ``_streaming_context`` variant used by
  ``forced_aligner.code2wav2aligner`` / ``minicpmo_4_5_omni.llm2tts`` is also
  recognized).
* C3 diffusion: ``(source_outputs, prompt, requires_multimodal_data,
  sampling_params=None)``.
* C4 legacy multi-source: ``(stage_list, engine_input_source, ...)`` — only
  ``moss_tts.talker2codec`` remains today; normalized by
  ``_adapt_moss_processor``, which binds ``stage_list=source_outputs`` and
  ``engine_input_source=ctx.prompt`` and drops the trailing
  ``prompt``/``requires_multimodal_data`` arguments (unused by that
  implementation).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vllm_omni.inputs.data import OmniTokensPrompt

__all__ = [
    "OrchestratorInputContext",
    "PlaceholderPromptBuilder",
    "DiffusionInputBuilder",
    "FullPayloadProducer",
    "AsyncChunkProducer",
    "wrap_orchestrator_processor",
    "invoke_orchestrator_processor",
]


@dataclass(frozen=True)
class OrchestratorInputContext:
    """Fixed transition context passed to orchestrator-facing processors.

    There is deliberately **no** ``model_config`` field: a processor that needs
    a model config reads it through the upstream stage closure, never through
    this context.
    """

    prompt: Any | None = None
    requires_multimodal_data: bool = False
    streaming_context: Any | None = None
    sampling_params: Any | None = None


@runtime_checkable
class PlaceholderPromptBuilder(Protocol):
    """C2-style sync builder: upstream outputs -> next-stage token prompts."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> list[OmniTokensPrompt]: ...


@runtime_checkable
class DiffusionInputBuilder(Protocol):
    """Diffusion-stage input builder: upstream outputs -> diffusion payload(s)."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> dict | list[dict] | None: ...


@runtime_checkable
class FullPayloadProducer(Protocol):
    """Producer-side (worker) full-payload builder.

    ``pooling_output`` is a **load-bearing** keyword name: it is the connector
    data-plane contract and must not be renamed or made positional.  These
    builders never receive an ``OrchestratorInputContext``.
    """

    def __call__(
        self,
        *,
        transfer_manager: Any,
        pooling_output: Any,
        request: Any,
        is_finished: bool = ...,
    ) -> Any: ...


@runtime_checkable
class AsyncChunkProducer(Protocol):
    """Producer-side (worker) async-chunk builder.

    ``multimodal_output`` is a **load-bearing** keyword name: it is the
    connector data-plane contract and must not be renamed or made positional.
    These builders never receive an ``OrchestratorInputContext``.
    """

    def __call__(
        self,
        *,
        transfer_manager: Any,
        multimodal_output: Any,
        request: Any,
        is_finished: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Signature inspection helpers
# ---------------------------------------------------------------------------


def _is_orchestrator_context_annotation(annotation: Any) -> bool:
    """Whether ``annotation`` names ``OrchestratorInputContext``.

    Handles the class object, plain string annotations (``from __future__
    import annotations``) and fully-qualified dotted strings.
    """
    if annotation is inspect.Parameter.empty:
        return False
    if annotation is OrchestratorInputContext:
        return True
    if isinstance(annotation, str):
        text = annotation.strip().strip("'\"")
        return text == "OrchestratorInputContext" or text.endswith("OrchestratorInputContext")
    return False


def _accepts_orchestrator_input_context(fn: Any) -> bool:
    """Whether ``fn`` already speaks the new ``ctx`` contract.

    True when the signature contains a parameter named ``ctx``, or any
    parameter whose type annotation is ``OrchestratorInputContext``.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for name, param in signature.parameters.items():
        if name == "ctx":
            return True
        if _is_orchestrator_context_annotation(param.annotation):
            return True
    return False


def _positional_parameter_names(fn: Any) -> list[str] | None:
    """Names of positional (or positional-or-keyword) parameters of ``fn``.

    Returns ``None`` when the signature cannot be inspected (e.g. builtins).
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            names.append(param.name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            names.append("*args")
    return names


def _warn_legacy_contract(fn: Any) -> None:
    """Emit a ``DeprecationWarning`` for a legacy positional processor shape."""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    warnings.warn(
        f"stage-input processor {name!r} uses the legacy positional contract; "
        "migrate it to the OrchestratorInputContext contract "
        "(RFC #4872 Phase 2).",
        DeprecationWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


def _adapt_moss_processor(fn: Any) -> Any:
    """C4 legacy multi-source adapter (e.g. ``moss_tts.talker2codec``).

    The legacy multi-source sync processors take
    ``(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False)``.
    The orchestrator contract dispatches ``(source_outputs, ctx)``; we bind
    ``stage_list=source_outputs`` and ``engine_input_source=ctx.prompt`` and
    drop the trailing ``prompt``/``requires_multimodal_data`` arguments (the
    ``moss_tts.talker2codec`` implementation does not use them, so this is
    behaviour-preserving w.r.t. the previous 4-arg positional call).
    """

    def _adapted(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
        return fn(source_outputs, ctx.prompt)

    _warn_legacy_contract(fn)
    return _adapted


def wrap_orchestrator_processor(fn: Any) -> PlaceholderPromptBuilder | DiffusionInputBuilder:
    """Return a C1-compatible callable ``(source_outputs, ctx)`` for ``fn``.

    - If ``fn`` already accepts ``ctx`` (``_accepts_orchestrator_input_context``),
      it is returned unchanged.
    - Otherwise the legacy positional shape is adapted:
      * C4 ``(stage_list, engine_input_source, ...)`` -> ``_adapt_moss_processor``;
      * C3 ``(source_outputs, prompt, requires_multimodal_data, sampling_params=...)``
        forwards ``ctx.sampling_params`` when the parameter exists;
      * C2 ``(source_outputs, prompt, requires_multimodal_data,
        streaming_context=...)`` (or ``_streaming_context``) forwards
        ``ctx.streaming_context``;
      * C0 3-arg ``(source_outputs, prompt, requires_multimodal_data)``.
    Every legacy shape emits a ``DeprecationWarning``.
    """
    if not callable(fn):
        raise TypeError(f"stage-input processor must be callable, got {fn!r}")

    if _accepts_orchestrator_input_context(fn):
        return fn  # type: ignore[return-value]

    names = _positional_parameter_names(fn)

    # Uninspectable callable (e.g. builtin): fall back to the most conservative
    # 3-arg C0 shape; the runtime previously treated it via the <4 arity path.
    if names is None:
        names = ["source_outputs", "prompt", "requires_multimodal_data"]

    # C4 legacy multi-source (e.g. moss_tts.talker2codec).
    if len(names) >= 2 and names[0] == "stage_list" and names[1] == "engine_input_source":
        return _adapt_moss_processor(fn)  # type: ignore[return-value]

    # C3 diffusion: forwarding sampling_params (diffusion-stage params).
    if "sampling_params" in names:

        def _c3(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
            return fn(source_outputs, ctx.prompt, ctx.requires_multimodal_data, sampling_params=ctx.sampling_params)

        _warn_legacy_contract(fn)
        return _c3  # type: ignore[return-value]

    # C2 placeholder: forwarding the streaming context.
    if "streaming_context" in names or "_streaming_context" in names:

        def _c2(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
            return fn(source_outputs, ctx.prompt, ctx.requires_multimodal_data, ctx.streaming_context)

        _warn_legacy_contract(fn)
        return _c2  # type: ignore[return-value]

    # C0 3-arg legacy.
    def _c0(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
        return fn(source_outputs, ctx.prompt, ctx.requires_multimodal_data)

    _warn_legacy_contract(fn)
    return _c0  # type: ignore[return-value]


def invoke_orchestrator_processor(
    fn: Any,
    source_outputs: list[Any],
    ctx: OrchestratorInputContext,
) -> Any:
    """Invoke an orchestrator-facing processor under the fixed ``(source_outputs, ctx)`` contract."""
    return wrap_orchestrator_processor(fn)(source_outputs, ctx)
