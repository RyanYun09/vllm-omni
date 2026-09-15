# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public model-runner interface for Omni connector transport."""

from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
    _OmniConnectorPayloadTransportMixin,
)
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_runtime import (
    _should_create_payload_connector,
    needs_omni_connector,
    should_accumulate_full_payload_output,
)
from vllm_omni.model_executor.stage_input_processors import (
    ProcessorValidationError,
    resolve_processor,
)
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.payload_span import (
    get_tensor_span,
    merge_tensor_spans,
)

__all__ = [
    "OmniConnectorModelRunnerMixin",
    "_should_create_payload_connector",
    "needs_omni_connector",
    "should_accumulate_full_payload_output",
]


class OmniConnectorModelRunnerMixin(_OmniConnectorPayloadTransportMixin):
    """Unified data-plane communication interface for model runners.

    Runtime ownership and payload transport live under
    "distributed.omni_connectors".
    """
