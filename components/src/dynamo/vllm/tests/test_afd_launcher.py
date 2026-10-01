# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check deployment isolation and failure cleanup without starting an engine."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    source = (
        Path(__file__).resolve().parents[5]
        / "examples/backends/vllm/launch/afd_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("afd_launcher", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "argv", [str(source)])
    for name in tuple(os.environ):
        if name.startswith(("AFD_", "DYN_", "MAX_", "MODEL", "VLLM_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AFD_LOG_DIR", str(tmp_path / "launch"))
    monkeypatch.setattr(module, "preflight", lambda: {"test": "no engine loaded"})
    monkeypatch.setattr(module, "require_free_local_ports", Mock())
    monkeypatch.setattr(module.signal, "signal", Mock())
    monkeypatch.setattr(module, "wait_until_ready", Mock())
    supervisor = module.Supervisor(tmp_path / "launch")
    supervisor.start = Mock()
    supervisor.wait = Mock(return_value=0)
    supervisor.close = Mock()
    monkeypatch.setattr(module, "Supervisor", lambda _: supervisor)
    return module, supervisor


@pytest.mark.parametrize("explicit_runtime", [False, True])
def test_launcher_isolates_discovery_and_keeps_ffn_off_request_plane(
    launcher, monkeypatch, tmp_path, explicit_runtime
):
    module, supervisor = launcher
    inherited = (
        "DYN_NAMESPACE_PREFIX",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_ENDPOINT",
        "DYN_KV_STATE_ENDPOINT",
    )
    for name in inherited:
        monkeypatch.setenv(name, "unrelated-deployment")
    if explicit_runtime:
        monkeypatch.setenv("DYN_DISCOVERY_BACKEND", "etcd")
        monkeypatch.setenv("DYN_EVENT_PLANE", "nats")
        monkeypatch.setenv("DYN_NAMESPACE", "chosen-experiment")

    assert module.main() == 0
    manifest = json.loads((tmp_path / "launch/launch.json").read_text())
    calls = {call.args[0]: call.args[1:] for call in supervisor.start.call_args_list}
    assert set(calls) == {"frontend", "ffn", "attention"}
    assert calls["ffn"][0][:2] == ["vllm", "serve"]
    assert calls["attention"][0][1:3] == ["-m", "dynamo.vllm"]
    assert not set(calls["ffn"][1]["CUDA_VISIBLE_DEVICES"].split(",")) & set(
        calls["attention"][1]["CUDA_VISIBLE_DEVICES"].split(",")
    )
    for _, env in calls.values():
        assert not any(name in env for name in inherited)
        assert env["DYN_NAMESPACE"] == manifest["dynamo_namespace"]
        assert env["DYN_DISCOVERY_BACKEND"] == ("etcd" if explicit_runtime else "file")
        assert env["DYN_EVENT_PLANE"] == ("nats" if explicit_runtime else "zmq")
        assert env["DYN_FILE_KV"] == str(tmp_path / "launch/discovery")
    if explicit_runtime:
        assert manifest["dynamo_namespace"] == "chosen-experiment"
    supervisor.close.assert_called_once()


def test_readiness_failure_cleans_up_all_started_roles(launcher):
    module, supervisor = launcher
    module.wait_until_ready.side_effect = TimeoutError("model never registered")
    assert module.main() == 1
    assert supervisor.start.call_count == 3
    supervisor.wait.assert_not_called()
    supervisor.close.assert_called_once()
