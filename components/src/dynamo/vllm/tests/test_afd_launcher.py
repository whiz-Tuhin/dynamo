# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check deployment isolation and failure cleanup without starting an engine."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
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
def launcher_module():
    source = (
        Path(__file__).resolve().parents[5]
        / "examples/backends/vllm/launch/afd_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("afd_launcher", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher(launcher_module, monkeypatch, tmp_path):
    module = launcher_module
    monkeypatch.setattr(module.sys, "argv", ["afd_launcher.py"])
    for name in tuple(os.environ):
        if name.startswith(("AFD_", "DYN_", "MAX_", "MODEL", "VLLM_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AFD_LOG_DIR", str(tmp_path / "launch"))
    monkeypatch.setattr(module, "preflight", lambda: {"test": "no engine loaded"})
    monkeypatch.setattr(module, "require_free_local_ports", Mock())
    monkeypatch.setattr(module.signal, "signal", Mock())
    monkeypatch.setattr(module, "wait_until_ready", Mock())
    monkeypatch.setattr(module, "wait_for_ffn_daemon", Mock())
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


@pytest.mark.parametrize("namespace", ["", " ", "dynamo"])
def test_global_namespace_is_rejected_before_starting_roles(
    launcher, monkeypatch, namespace
):
    module, supervisor = launcher
    monkeypatch.setenv("DYN_NAMESPACE", namespace)
    with pytest.raises(ValueError, match="DYN_NAMESPACE"):
        module.main()
    supervisor.start.assert_not_called()


def test_missing_ffn_daemon_evidence_fails_closed(launcher):
    module, supervisor = launcher
    module.wait_for_ffn_daemon.side_effect = TimeoutError("missing FFN daemon")
    assert module.main() == 1
    supervisor.wait.assert_not_called()
    supervisor.close.assert_called_once()


@pytest.mark.parametrize("is_afd", [False, True])
def test_ffn_log_gate_requires_actual_connector_loop_marker(
    launcher_module, tmp_path, is_afd
):
    log = "Application startup complete."
    if is_afd:
        log += "\nAFD FFN EngineCore started; workers run connector loop."
    (tmp_path / "ffn.log").write_text(log)
    supervisor = SimpleNamespace(
        log_dir=tmp_path, signal_number=None, exited_child=lambda: None
    )
    if is_afd:
        launcher_module.wait_for_ffn_daemon(supervisor, timeout=1)
    else:
        with pytest.raises(TimeoutError, match="connector-loop startup evidence"):
            launcher_module.wait_for_ffn_daemon(supervisor, timeout=0.01)


def test_preflight_exposes_compatibility_error_hidden_by_entry_point(
    launcher_module, monkeypatch
):
    module = launcher_module
    plugin = SimpleNamespace(register_afd=Mock(), __file__=__file__)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.26.0")
    monkeypatch.setattr(
        module.importlib.metadata,
        "entry_points",
        lambda **_: [SimpleNamespace(name="afd", value="afd_plugin:register_afd")],
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: None),
    )

    def import_module(name):
        if name == "afd_plugin":
            return plugin
        raise ImportError("incompatible vLLM patch API")

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    with pytest.raises(ImportError, match="incompatible vLLM patch API"):
        module.preflight()
    plugin.register_afd.assert_called_once()
