# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo integration boundary for the external vLLM AFD plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynamo.vllm.constants import DisaggregationMode


def validate_afd_sidecar_contract(
    disaggregation_mode: DisaggregationMode,
    additional_config: Mapping[str, Any] | None,
) -> None:
    """Validate the request-facing half of an AFD sidecar deployment.

    The AFD plugin's Attention role owns scheduling, KV cache, sampling, and the
    request lifecycle, so it can run inside ``dynamo.vllm``. The FFN role is a
    connector-driven daemon and must run as a native ``vllm serve`` sidecar; it
    cannot register a Dynamo request endpoint. In a combined PD+AFD deployment,
    only the decode worker uses the AFD Attention role.
    """

    if additional_config is None:
        return

    afd_config = additional_config.get("afd")
    if afd_config is None:
        return
    if not isinstance(afd_config, Mapping):
        raise TypeError("vLLM additional_config['afd'] must be an object")

    role = afd_config.get("role", "attention")
    if role == "ffn":
        raise ValueError(
            "AFD role 'ffn' is connector-driven and cannot run as a Dynamo "
            "request worker; launch it with native `vllm serve` as a sidecar"
        )
    if role != "attention":
        raise ValueError("AFD role must be 'attention' or 'ffn'")

    if disaggregation_mode in {
        DisaggregationMode.PREFILL,
        DisaggregationMode.ENCODE,
    }:
        raise ValueError(
            "AFD Attention is supported only for aggregated or decode Dynamo "
            "workers; keep PD prefill and multimodal encode workers aggregated"
        )
