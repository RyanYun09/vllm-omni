# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the RFC #4872 Phase 3 (Part A) processor registry.

Pure-logic, CPU-only: exercises kind inference, structural validation,
dead-processor hints and ``resolve_processor`` without executing any processor
logic or loading model weights.

Subject under test: ``vllm_omni.model_executor.stage_input_processors._registry``
"""

import importlib
import warnings

import pytest

from vllm_omni.model_executor.stage_input_processors import (
    ProcessorKind,
    ProcessorSpec,
    ProcessorValidationError,
    _registry,
    dead_processor_hint,
    infer_kind,
    register_processor,
    resolve_processor,
    validate_processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Dummy processors (simple callables used as validation subjects).
# ---------------------------------------------------------------------------


def placeholder_a_token_only(source_outputs, _prompt=None, _requires_multimodal_data=False):
    return []


def placeholder_b_token_only(prompt, ctx=None):
    return []


def placeholder_c_badname(source_outputs, _prompt=None):
    return []


def diffusion_a_ar2diffusion(source_outputs, _prompt=None, _requires_multimodal_data=False):
    return {}


def producer_a_full_payload(transfer_manager, pooling_output, request):
    return None


def producer_b_full_payload(transfer_manager, pooling_output, request, is_finished=False):
    return None


def producer_c_full_payload(transfer_manager, multimodal_output, request, is_finished=False):
    return None


def producer_a_async_chunk(transfer_manager, multimodal_output, request, is_finished=False):
    return None


def producer_b_async_chunk(transfer_manager, multimodal_output, request):
    return None


def producer_c_async_chunk(transfer_manager, multimodal_output, request, is_finished, /):
    return None


def talker2codec_shape(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
    return []


def _mod_path(fn_name: str) -> str:
    """Dotted path to a callable defined in this test module."""
    return f"{__name__}.{fn_name}"


# ---------------------------------------------------------------------------
# Kind inference (suffix rules).
# ---------------------------------------------------------------------------


class TestInferKind:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("pkg.mod.foo_token_only", "placeholder_prompt_builder"),
            ("pkg.mod.foo_full_payload", "producer_full_payload"),
            ("pkg.mod.foo_batch", "producer_full_payload"),
            ("pkg.mod.foo_async_chunk", "producer_async_chunk"),
            ("pkg.mod.ar2diffusion", "diffusion_input_builder"),
            ("pkg.mod.ar2dit", "diffusion_input_builder"),
            ("pkg.mod.thinker2imagegen", "diffusion_input_builder"),
            ("pkg.mod.legacy_builder", "legacy_orchestrator_builder"),
            ("pkg.mod.talker2code2wav", "legacy_orchestrator_builder"),
            ("vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec", "legacy_multi_source"),
            ("pkg.moss_tts.talker2codec", "legacy_multi_source"),
            (
                "vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec_delay_async_chunk",
                "producer_async_chunk",
            ),
        ],
    )
    def test_suffix_rules(self, path: str, expected: ProcessorKind):
        assert infer_kind(None, path=path) == expected

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            infer_kind(None, path="")
        with pytest.raises(ValueError):
            infer_kind(None, path="nodots")

    def test_register_processor_override(self):
        register_processor("pkg.mod.custom_path", "diffusion_input_builder")
        try:
            assert infer_kind(None, path="pkg.mod.custom_path") == "diffusion_input_builder"
        finally:
            _registry._KIND_OVERRIDES.clear()

    def test_register_processor_invalid_kind(self):
        with pytest.raises(ValueError):
            register_processor("pkg.mod.custom_path", "bogus_kind")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Structural validation.
# ---------------------------------------------------------------------------


class TestValidateProcessor:
    def test_full_payload_ok_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                producer_b_full_payload,
                kind="producer_full_payload",
                path="pkg.mod.producer_b_full_payload",
            )

    def test_full_payload_missing_is_finished_warns(self):
        with pytest.warns(RuntimeWarning, match="is_finished"):
            validate_processor(
                producer_a_full_payload,
                kind="producer_full_payload",
                path="pkg.mod.producer_a_full_payload",
            )

    def test_full_payload_cross_kind_second_param_warns(self):
        with pytest.warns(RuntimeWarning, match="multimodal_output"):
            validate_processor(
                producer_c_full_payload,
                kind="producer_full_payload",
                path="pkg.mod.producer_c_full_payload",
            )

    def test_full_payload_missing_transfer_manager_fails(self):
        def bad(manager, pooling_output, request):
            return None

        with pytest.raises(ProcessorValidationError) as excinfo:
            validate_processor(bad, kind="producer_full_payload", path="pkg.mod.producer_full_payload")
        assert excinfo.value.rule == "transfer_manager"

    def test_async_chunk_ok_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                producer_a_async_chunk,
                kind="producer_async_chunk",
                path="pkg.mod.producer_a_async_chunk",
            )

    def test_async_chunk_missing_is_finished_hard_fail(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            validate_processor(
                producer_b_async_chunk,
                kind="producer_async_chunk",
                path="pkg.mod.producer_b_async_chunk",
            )
        assert excinfo.value.rule == "is_finished_required"
        assert "producer_b_async_chunk" in str(excinfo.value)

    def test_async_chunk_positional_only_is_finished_hard_fail(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            validate_processor(
                producer_c_async_chunk,
                kind="producer_async_chunk",
                path="pkg.mod.producer_c_async_chunk",
            )
        assert excinfo.value.rule == "is_finished_positional_only"

    def test_placeholder_ok_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                placeholder_a_token_only,
                kind="placeholder_prompt_builder",
                path="pkg.mod.placeholder_a_token_only",
            )

    def test_placeholder_ctx_contract_ok(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                placeholder_b_token_only,
                kind="placeholder_prompt_builder",
                path="pkg.mod.placeholder_b_token_only",
            )

    def test_placeholder_suffix_kind_conflict_warns(self):
        with pytest.warns(RuntimeWarning, match="suffix"):
            validate_processor(
                placeholder_c_badname,
                kind="placeholder_prompt_builder",
                path="pkg.mod.placeholder_c_badname",
            )

    def test_diffusion_ok_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                diffusion_a_ar2diffusion,
                kind="diffusion_input_builder",
                path="pkg.mod.diffusion_a_ar2diffusion",
            )

    def test_legacy_multi_source_allowlist_ok(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_processor(
                talker2codec_shape,
                kind="legacy_multi_source",
                path="vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec",
            )

    def test_legacy_multi_source_allowlist_rejects_non_moss(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            validate_processor(
                talker2codec_shape,
                kind="legacy_multi_source",
                path="pkg.mod.talker2codec",
            )
        assert excinfo.value.rule == "legacy_multi_source_allowlist"

    def test_uninspectable_callable_fails(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            validate_processor(
                object(),  # not callable / not inspectable
                kind="producer_full_payload",
                path="pkg.mod.producer_full_payload",
            )
        assert excinfo.value.rule == "inspectable_signature"


# ---------------------------------------------------------------------------
# Dead-processor hint (gate three-state).
# ---------------------------------------------------------------------------


class TestDeadProcessor:
    def test_async_chunk_receiving_downstream_is_dead(self):
        assert (
            dead_processor_hint(
                "placeholder_prompt_builder",
                async_chunk=True,
                downstream_receives_async_chunks=True,
                has_sync=False,
            )
            is True
        )

    def test_async_chunk_non_receiving_downstream_alive(self):
        assert (
            dead_processor_hint(
                "diffusion_input_builder",
                async_chunk=True,
                downstream_receives_async_chunks=False,
                has_sync=False,
            )
            is False
        )

    def test_sync_mode_alive_unless_overridden(self):
        assert (
            dead_processor_hint(
                "diffusion_input_builder",
                async_chunk=False,
                downstream_receives_async_chunks=False,
                has_sync=False,
            )
            is False
        )
        assert (
            dead_processor_hint(
                "diffusion_input_builder",
                async_chunk=False,
                downstream_receives_async_chunks=False,
                has_sync=True,
            )
            is True
        )

    def test_producer_never_dead(self):
        for kind in ("producer_full_payload", "producer_async_chunk"):
            assert (
                dead_processor_hint(
                    kind,
                    async_chunk=True,
                    downstream_receives_async_chunks=True,
                    has_sync=False,
                )
                is False
            )


# ---------------------------------------------------------------------------
# Resolution entry point.
# ---------------------------------------------------------------------------


class TestResolveProcessor:
    def test_resolve_normal(self):
        spec = resolve_processor(
            _mod_path("placeholder_a_token_only"),
            expected_kind="placeholder_prompt_builder",
        )
        assert isinstance(spec, ProcessorSpec)
        assert spec.kind == "placeholder_prompt_builder"
        assert spec.path == _mod_path("placeholder_a_token_only")
        assert spec.fn is placeholder_a_token_only

    def test_resolve_full_payload_without_expected_kind(self):
        spec = resolve_processor(_mod_path("producer_b_full_payload"))
        assert spec.kind == "producer_full_payload"

    def test_resolve_drop_in_equivalence(self):
        """resolve_processor().fn must be the same object as the legacy lookup."""
        spec = resolve_processor(
            _mod_path("producer_b_full_payload"),
            expected_kind="producer_full_payload",
        )
        legacy = getattr(importlib.import_module(__name__), "producer_b_full_payload")
        assert spec.fn is legacy

    def test_expected_kind_mismatch_raises(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            resolve_processor(
                _mod_path("placeholder_a_token_only"),
                expected_kind="producer_full_payload",
            )
        assert excinfo.value.rule == "expected_kind"
        assert "placeholder_prompt_builder" in str(excinfo.value)

    def test_missing_symbol_raises(self):
        with pytest.raises(AttributeError):
            resolve_processor(_mod_path("does_not_exist"))

    def test_package_exports(self):
        import vllm_omni.model_executor.stage_input_processors as sip

        assert sip.resolve_processor is resolve_processor
        assert sip.ProcessorSpec is ProcessorSpec
        assert sip.ProcessorValidationError is ProcessorValidationError
        assert sip.infer_kind is infer_kind
        assert sip.dead_processor_hint is dead_processor_hint
        assert sip.register_processor is register_processor
