# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the equal-budget AFD comparison plans."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[5]
    / "examples/backends/vllm/launch/afd_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("afd_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARISON
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC.loader.exec_module(COMPARISON)
sys.path.pop(0)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def make_plan(tmp_path: Path, arm: str, *, enable_dbo: bool = False):
    settings = COMPARISON.Settings(
        arm=arm,
        model=COMPARISON.DEFAULT_MODEL,
        revision=COMPARISON.DEFAULT_REVISION,
        log_dir=tmp_path / arm,
        namespace=f"test-{arm}",
        port_base=24000,
        enable_dbo=enable_dbo,
    )
    return COMPARISON.build_plan(settings, source_env={})


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("namespace", ["", " ", "dynamo"])
def test_shared_or_empty_namespace_is_rejected(tmp_path, namespace):
    settings = COMPARISON.Settings(
        arm="agg",
        model=COMPARISON.DEFAULT_MODEL,
        revision=COMPARISON.DEFAULT_REVISION,
        log_dir=tmp_path / "agg",
        namespace=namespace,
        port_base=24000,
    )
    with pytest.raises(ValueError, match="namespace"):
        COMPARISON.build_plan(settings, source_env={})


@pytest.mark.parametrize("arm", ["agg", "pd", "afd", "pd-afd"])
def test_each_arm_owns_exactly_four_disjoint_devices(tmp_path, arm):
    plan = make_plan(tmp_path, arm)
    devices = [device for process in plan.processes for device in process.devices]

    assert sorted(devices) == ["0", "1", "2", "3"]
    assert len(plan.ports) == len(set(plan.ports.values()))
    assert all(
        process.environment["DYN_NAMESPACE"] == f"test-{arm}"
        for process in plan.processes
    )


def test_aggregated_uses_dp4_tp1_with_expert_parallel(tmp_path):
    plan = make_plan(tmp_path, "agg")
    worker = next(process for process in plan.processes if process.role == "aggregated")

    assert command_value(worker.command, "--data-parallel-size") == "4"
    assert command_value(worker.command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in worker.command
    assert "--disaggregation-mode" not in worker.command
    assert worker.environment["VLLM_PLUGINS"] == ""


def test_pd_matches_pd_afd_prefill_footprint(tmp_path):
    plan = make_plan(tmp_path, "pd")
    prefill = [process for process in plan.processes if process.role == "prefill"]
    decode = [process for process in plan.processes if process.role == "decode"]

    assert len(prefill) == 2
    assert len(decode) == 2
    combined = make_plan(tmp_path, "pd-afd")
    assert {p.devices for p in prefill} == {
        p.devices for p in combined.processes if p.role == "prefill"
    }
    assert command_value(prefill[0].command, "--disaggregation-mode") == "prefill"
    assert all(
        command_value(process.command, "--disaggregation-mode") == "decode"
        for process in decode
    )
    assert all(
        command_value(process.command, "--kv-transfer-config")
        == COMPARISON.KV_TRANSFER_CONFIG
        for process in prefill + decode
    )
    assert (
        len(
            {
                process.environment["VLLM_NIXL_SIDE_CHANNEL_PORT"]
                for process in prefill + decode
            }
        )
        == 4
    )
    assert all(
        process.environment["UCX_TLS"] == "cuda_ipc,cuda_copy,tcp"
        for process in prefill + decode
    )


def test_pd_afd_routes_nixl_only_to_prefill_and_decode_attention(tmp_path):
    plan = make_plan(tmp_path, "pd-afd")
    prefill = [process for process in plan.processes if process.role == "prefill"]
    attention = next(
        process for process in plan.processes if process.role == "decode-attention"
    )
    ffn = next(process for process in plan.processes if process.role == "ffn")

    assert len(prefill) == 2
    assert command_value(attention.command, "--disaggregation-mode") == "decode"
    attention_config = json_value(attention.command, "--additional-config")
    ffn_config = json_value(ffn.command, "--additional-config")
    assert attention_config["afd"]["role"] == "attention"
    assert ffn_config["afd"]["role"] == "ffn"
    assert (
        command_value(attention.command, "--worker-cls")
        == "afd_plugin.v1.worker.AFDAttentionWorker"
    )
    assert (
        command_value(ffn.command, "--worker-cls")
        == "afd_plugin.v1.worker.AFDFFNWorker"
    )
    assert "--kv-transfer-config" in attention.command
    assert "--kv-transfer-config" not in ffn.command
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in ffn.environment
    assert {process.devices for process in prefill} == {("0",), ("1",)}
    assert attention.devices == ("2",)
    assert ffn.devices == ("3",)


def json_value(command: list[str], flag: str):
    return json.loads(command_value(command, flag))


@pytest.mark.parametrize("arm", ["agg", "pd", "afd", "pd-afd"])
def test_engine_commands_share_model_revision_dtype_and_eager_mode(tmp_path, arm):
    plan = make_plan(tmp_path, arm)
    engines = [process for process in plan.processes if process.role != "frontend"]

    for process in engines:
        assert (
            command_value(process.command, "--revision") == COMPARISON.DEFAULT_REVISION
        )
        assert (
            command_value(process.command, "--tokenizer-revision")
            == COMPARISON.DEFAULT_REVISION
        )
        assert command_value(process.command, "--dtype") == "bfloat16"
        assert "--enable-expert-parallel" in process.command
        assert "--no-enable-prefix-caching" in process.command
        assert "--enforce-eager" in process.command
        assert "--enable-dbo" not in process.command


@pytest.mark.parametrize("arm", ["agg", "pd", "afd", "pd-afd"])
def test_unqualified_matched_dbo_is_rejected(tmp_path, arm):
    with pytest.raises(ValueError, match="native workers require"):
        make_plan(tmp_path, arm, enable_dbo=True)


def test_afd_builds_direct_guarded_2a2f_roles(tmp_path):
    plan = make_plan(tmp_path, "afd")
    attention = next(
        process for process in plan.processes if process.role == "attention"
    )
    ffn = next(process for process in plan.processes if process.role == "ffn")

    assert attention.devices == ("0", "1")
    assert ffn.devices == ("2", "3")
    assert command_value(attention.command, "--data-parallel-size") == "1"
    assert command_value(attention.command, "--tensor-parallel-size") == "2"
    assert command_value(ffn.command, "--data-parallel-size") == "1"
    assert command_value(ffn.command, "--tensor-parallel-size") == "2"
    assert (
        command_value(attention.command, "--worker-cls")
        == "afd_plugin.v1.worker.AFDAttentionWorker"
    )
    assert (
        command_value(ffn.command, "--worker-cls")
        == "afd_plugin.v1.worker.AFDFFNWorker"
    )
    assert "--enable-dbo" not in attention.command
    assert "--enable-dbo" not in ffn.command


def test_pd_allocation_tuning_keeps_four_device_budget(tmp_path):
    settings = COMPARISON.Settings(
        arm="pd",
        model=COMPARISON.DEFAULT_MODEL,
        revision=COMPARISON.DEFAULT_REVISION,
        log_dir=tmp_path / "pd",
        namespace="test-pd",
        port_base=24000,
        pd_prefill_workers=1,
    )
    plan = COMPARISON.build_plan(settings, source_env={})
    assert len([p for p in plan.processes if p.role == "prefill"]) == 1
    assert len([p for p in plan.processes if p.role == "decode"]) == 3
    assert sorted(d for p in plan.processes for d in p.devices) == ["0", "1", "2", "3"]
