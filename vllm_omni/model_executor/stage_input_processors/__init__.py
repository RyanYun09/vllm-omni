"""Stage input processors: per-model builders for inter-stage transitions.

Module-naming convention (RFC #4872):
    - ``*_full_payload`` : producer-side (worker) connector payload builder
      (``FullPayloadProducer``; kwarg ``pooling_output`` load-bearing).
    - ``*_async_chunk``  : producer-side (worker) streaming chunk builder
      (``AsyncChunkProducer``; kwarg ``multimodal_output`` load-bearing).
    - ``*_token_only``   : consumer-side (orchestrator) length-only placeholder
      builder for the non-async-chunk path.
    - no suffix (legacy) : consumer-side (orchestrator) sync builder; may be a
      placeholder builder or a diffusion input builder.

Orchestrator-facing roles are normalized through
``wrap_orchestrator_processor`` / ``invoke_orchestrator_processor``:
    - placeholder : upstream outputs -> next-stage token prompts
      (``PlaceholderPromptBuilder``).
    - diffusion   : upstream outputs -> diffusion payload(s)
      (``DiffusionInputBuilder``).

This package only re-exports the dispatch contract; it imports no other
processor modules, so importing it never pulls in model-specific code.
"""

from vllm_omni.model_executor.stage_input_processors._dispatch import (
    AsyncChunkProducer,
    DiffusionInputBuilder,
    FullPayloadProducer,
    OrchestratorInputContext,
    PlaceholderPromptBuilder,
    invoke_orchestrator_processor,
    wrap_orchestrator_processor,
)

__all__ = [
    "OrchestratorInputContext",
    "PlaceholderPromptBuilder",
    "DiffusionInputBuilder",
    "FullPayloadProducer",
    "AsyncChunkProducer",
    "invoke_orchestrator_processor",
    "wrap_orchestrator_processor",
]
