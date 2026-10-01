#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supervise a single-node Dynamo Attention + native vLLM FFN deployment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shlex
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from dynamo.vllm.afd_supervisor import Supervisor

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V2-Lite"
DEFAULT_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
REQUIRED_VLLM = "0.26.0"


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def parse_devices(name: str, default: str) -> list[str]:
    devices = [item.strip() for item in os.environ.get(name, default).split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError(f"{name} must be a comma-separated, non-empty device list")
    if len(set(devices)) != len(devices):
        raise ValueError(f"{name} contains duplicate devices")
    return devices


def preflight() -> dict[str, object]:
    versions: dict[str, object] = {}
    for package in ("vllm", "vllm-afd-plugin"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required package is not installed: {package}"
            ) from error
    if versions["vllm"] != REQUIRED_VLLM:
        raise RuntimeError(
            f"vLLM {REQUIRED_VLLM} is required by this AFD pin; "
            f"found {versions['vllm']}"
        )
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    if not any(
        ep.name == "afd" and ep.value == "afd_plugin:register_afd"
        for ep in entry_points
    ):
        raise RuntimeError(
            "vLLM plugin entry point 'afd = afd_plugin:register_afd' is missing"
        )
    plugin_module = importlib.import_module("afd_plugin")
    plugin_module.register_afd()
    # The pinned plugin suppresses compatibility-import errors in its entry
    # point. Import explicitly and check installed functions so a native engine
    # cannot silently stand in for an AFD role.
    patches = {}
    for name in (
        "async_dp_engine",
        "async_dp_forward_context",
        "config_validation",
        "engine_core",
    ):
        patches[name] = importlib.import_module(f"afd_plugin.compat.patches.{name}")
    config_patch = patches["config_validation"]
    engine_patch = patches["engine_core"]
    if (
        config_patch.arg_utils_module.EngineArgs.create_engine_config
        is not config_patch.create_engine_config
        or config_patch.config_module.VllmConfig.__post_init__
        is not config_patch.__post_init__
        or engine_patch.core_module.EngineCore.__init__ is not engine_patch.__init__
        or engine_patch.core_module.EngineCoreProc.run_busy_loop
        is not engine_patch.run_busy_loop
        or engine_patch.core_module.DPEngineCoreProc.run_busy_loop
        is not engine_patch.run_busy_loop
    ):
        raise RuntimeError("required AFD compatibility patches are not installed")
    versions["afd_verified_patches"] = list(patches)
    distribution = importlib.metadata.distribution("vllm-afd-plugin")
    direct_url = distribution.read_text("direct_url.json")
    versions["afd_plugin_module"] = str(Path(plugin_module.__file__).resolve())
    versions["afd_plugin_direct_url"] = (
        json.loads(direct_url) if direct_url is not None else None
    )
    return versions


def require_free_local_ports(ports: dict[str, int]) -> None:
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"{name}={port} is unavailable on localhost"
                ) from error


def wait_until_ready(
    supervisor: Supervisor,
    url: str,
    expected_model: str,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if supervisor.signal_number is not None:
            raise KeyboardInterrupt
        exited = supervisor.exited_child()
        if exited is not None:
            raise RuntimeError(
                f"{exited.name} exited with {exited.process.returncode}; "
                f"see {exited.log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
                model_ids = {
                    item.get("id")
                    for item in payload.get("data", [])
                    if isinstance(item, dict)
                }
                if response.status == 200 and expected_model in model_ids:
                    print(
                        f"frontend registered: {url} reports {expected_model}",
                        flush=True,
                    )
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError(
        f"deployment did not report model {expected_model!r} within {timeout}s: {url}"
    )


def afd_config(
    role: str,
    host: str,
    port: int,
    attention_ranks: int,
    ffn_ranks: int,
) -> str:
    return json.dumps(
        {
            "afd": {
                "role": role,
                "connector": "P2pNcclAFDConnector",
                "host": host,
                "port": port,
                "num_attention_ranks": attention_ranks,
                "num_ffn_ranks": ffn_ranks,
            }
        },
        separators=(",", ":"),
    )


def wait_for_ffn_daemon(supervisor: Supervisor, timeout: int = 30) -> None:
    """Require the connector loop's startup acknowledgement, not FFN HTTP."""
    deadline = time.monotonic() + timeout
    log_path = supervisor.log_dir / "ffn.log"
    while time.monotonic() < deadline:
        if supervisor.signal_number is not None:
            raise KeyboardInterrupt
        if supervisor.exited_child() is not None:
            raise RuntimeError("AFD role exited before FFN connector-loop readiness")
        if log_path.exists() and (
            "AFD FFN EngineCore started; workers run connector loop."
            in log_path.read_text(errors="replace")
        ):
            return
        time.sleep(0.2)
    raise TimeoutError(f"missing AFD FFN connector-loop startup evidence: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch experimental AFD: Dynamo Attention + native vLLM FFN sidecar."
        )
    )
    parser.parse_args()
    versions = preflight()

    model = os.environ.get("MODEL", DEFAULT_MODEL)
    revision = os.environ.get("MODEL_REVISION", DEFAULT_REVISION)
    attention_devices = parse_devices("AFD_ATTENTION_DEVICES", "0,1")
    ffn_devices = parse_devices("AFD_FFN_DEVICES", "2,3")
    overlap = set(attention_devices) & set(ffn_devices)
    if overlap:
        raise ValueError(f"Attention and FFN devices overlap: {sorted(overlap)}")

    attention_dp = env_int("AFD_ATTENTION_DP", 1)
    attention_tp = env_int("AFD_ATTENTION_TP", 2)
    ffn_dp = env_int("AFD_FFN_DP", 1)
    ffn_tp = env_int("AFD_FFN_TP", 2)
    attention_ranks = attention_dp * attention_tp
    ffn_ranks = ffn_dp * ffn_tp
    if len(attention_devices) != attention_ranks:
        raise ValueError(
            "AFD_ATTENTION_DEVICES count must equal AFD_ATTENTION_DP * AFD_ATTENTION_TP"
        )
    if len(ffn_devices) != ffn_ranks:
        raise ValueError("AFD_FFN_DEVICES count must equal AFD_FFN_DP * AFD_FFN_TP")
    if attention_ranks < ffn_ranks:
        raise ValueError("P2pNcclAFDConnector requires Attention ranks >= FFN ranks")

    afd_host = os.environ.get("AFD_HOST", "127.0.0.1")
    afd_port = env_int("AFD_PORT", 6269)
    ffn_port = env_int("AFD_FFN_HTTP_PORT", 18001)
    http_port = env_int("DYN_HTTP_PORT", 8000)
    system_port = env_int("DYN_SYSTEM_PORT", 8081)
    ports = {
        "AFD_PORT": afd_port,
        "AFD_FFN_HTTP_PORT": ffn_port,
        "DYN_HTTP_PORT": http_port,
        "DYN_SYSTEM_PORT": system_port,
    }
    if any(port > 65535 for port in ports.values()):
        raise ValueError("all ports must be <= 65535")
    if len(set(ports.values())) != len(ports):
        raise ValueError(
            f"AFD, FFN, frontend, and system ports must be distinct: {ports}"
        )
    require_free_local_ports(ports)

    ready_timeout = env_int("AFD_READY_TIMEOUT", 900)
    log_dir = Path(os.environ.get("AFD_LOG_DIR", f"/tmp/dynamo-afd-{os.getpid()}"))
    max_model_len = env_int("MAX_MODEL_LEN", 8192)
    max_seqs = env_int("MAX_CONCURRENT_SEQS", 64)
    max_tokens = env_int("MAX_BATCHED_TOKENS", 64)
    enable_dbo = env_flag("AFD_ENABLE_DBO")

    common = [
        "--revision",
        revision,
        "--tokenizer-revision",
        revision,
        "--dtype",
        "bfloat16",
        "--enable-expert-parallel",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        str(max_seqs),
        "--max-num-batched-tokens",
        str(max_tokens),
        "--enforce-eager",
        "--trust-remote-code",
    ]
    if enable_dbo:
        common.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                "2",
                "--dbo-prefill-token-threshold",
                "12",
            ]
        )
    ffn_command = [
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(ffn_port),
        "--data-parallel-size",
        str(ffn_dp),
        "--tensor-parallel-size",
        str(ffn_tp),
        "--additional-config",
        afd_config("ffn", afd_host, afd_port, attention_ranks, ffn_ranks),
        "--worker-cls",
        "afd_plugin.v1.worker.AFDFFNWorker",
        *common,
    ]
    attention_command = [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        model,
        "--data-parallel-size",
        str(attention_dp),
        "--tensor-parallel-size",
        str(attention_tp),
        "--additional-config",
        afd_config("attention", afd_host, afd_port, attention_ranks, ffn_ranks),
        "--worker-cls",
        "afd_plugin.v1.worker.AFDAttentionWorker",
        "--max-model-len",
        str(max_model_len),
        *common,
    ]
    attention_command.extend(shlex.split(os.environ.get("AFD_VLLM_GPU_MEM_ARGS", "")))

    base_env = os.environ.copy()
    # Inherited broad discovery filters or explicit endpoints must not redirect
    # this standalone experiment into another deployment's namespace.
    for key in (
        "DYN_NAMESPACE_PREFIX",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_ENDPOINT",
        "DYN_KV_STATE_ENDPOINT",
    ):
        base_env.pop(key, None)
    namespace = os.environ.get("DYN_NAMESPACE", f"dynamo-afd-{uuid.uuid4().hex}")
    if not namespace.strip() or namespace == "dynamo":
        raise ValueError("DYN_NAMESPACE must be nonempty and cannot be global 'dynamo'")
    base_env.update(
        {
            "DYN_NAMESPACE": namespace,
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_PLUGINS": "afd",
            "PYTHONUNBUFFERED": "1",
        }
    )
    base_env.setdefault("DYN_DISCOVERY_BACKEND", "file")
    base_env.setdefault("DYN_REQUEST_PLANE", "tcp")
    base_env.setdefault("DYN_EVENT_PLANE", "zmq")
    base_env.setdefault("DYN_FILE_KV", str(log_dir.resolve() / "discovery"))
    frontend_env = base_env.copy()
    ffn_env = base_env | {"CUDA_VISIBLE_DEVICES": ",".join(ffn_devices)}
    attention_env = base_env | {
        "CUDA_VISIBLE_DEVICES": ",".join(attention_devices),
        "DYN_SYSTEM_PORT": str(system_port),
    }

    supervisor = Supervisor(log_dir)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, supervisor.handle_signal)
    manifest = {
        "model": model,
        "model_revision": revision,
        "dtype": "bfloat16",
        "enable_dbo": enable_dbo,
        "dynamo_namespace": namespace,
        "runtime": {
            key: base_env[key]
            for key in (
                "DYN_DISCOVERY_BACKEND",
                "DYN_REQUEST_PLANE",
                "DYN_EVENT_PLANE",
                "DYN_FILE_KV",
            )
        },
        "versions": versions,
        "attention_devices": attention_devices,
        "ffn_devices": ffn_devices,
        "attention_command": attention_command,
        "ffn_command": ffn_command,
        "ports": ports,
    }
    (log_dir / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"model={model}@{revision}\nlogs={log_dir}", flush=True)
    try:
        supervisor.start(
            "frontend",
            [sys.executable, "-m", "dynamo.frontend"],
            frontend_env,
        )
        supervisor.start("ffn", ffn_command, ffn_env)
        supervisor.start("attention", attention_command, attention_env)
        frontend_host = os.environ.get("DYN_HTTP_HOST", "127.0.0.1")
        if frontend_host in {"0.0.0.0", "::"}:
            frontend_host = "127.0.0.1"
        wait_until_ready(
            supervisor,
            f"http://{frontend_host}:{http_port}/v1/models",
            model,
            ready_timeout,
        )
        wait_for_ffn_daemon(supervisor)
        print("ready: Dynamo Attention and AFD FFN connector loop", flush=True)
        return supervisor.wait()
    except KeyboardInterrupt:
        return 128 + (supervisor.signal_number or signal.SIGINT)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"AFD launch failed: {error}", file=sys.stderr)
        return 1
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
