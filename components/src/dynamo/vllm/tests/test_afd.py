# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the external vLLM AFD plugin integration boundary."""

import pytest
from dynamo.vllm.afd import validate_afd_sidecar_contract
from dynamo.vllm.constants import DisaggregationMode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "mode",
    [DisaggregationMode.AGGREGATED, DisaggregationMode.DECODE],
)
def test_attention_role_is_request_facing(mode):
    validate_afd_sidecar_contract(
        mode,
        {"afd": {"role": "attention", "connector": "P2pNcclAFDConnector"}},
    )


def test_missing_afd_config_keeps_existing_worker_behavior():
    validate_afd_sidecar_contract(DisaggregationMode.AGGREGATED, None)
    validate_afd_sidecar_contract(DisaggregationMode.AGGREGATED, {})


def test_default_afd_role_is_attention():
    validate_afd_sidecar_contract(
        DisaggregationMode.AGGREGATED,
        {"afd": {"connector": "P2pNcclAFDConnector"}},
    )


def test_ffn_role_requires_native_vllm_sidecar():
    with pytest.raises(ValueError, match="connector-driven"):
        validate_afd_sidecar_contract(
            DisaggregationMode.AGGREGATED,
            {"afd": {"role": "ffn"}},
        )


@pytest.mark.parametrize(
    "mode",
    [DisaggregationMode.PREFILL, DisaggregationMode.ENCODE],
)
def test_attention_role_rejects_non_request_facing_dynamo_stages(mode):
    with pytest.raises(ValueError, match="aggregated or decode"):
        validate_afd_sidecar_contract(mode, {"afd": {"role": "attention"}})


def test_afd_config_must_be_an_object():
    with pytest.raises(TypeError, match="must be an object"):
        validate_afd_sidecar_contract(
            DisaggregationMode.AGGREGATED,
            {"afd": "attention"},
        )


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="must be 'attention' or 'ffn'"):
        validate_afd_sidecar_contract(
            DisaggregationMode.AGGREGATED,
            {"afd": {"role": "router"}},
        )
