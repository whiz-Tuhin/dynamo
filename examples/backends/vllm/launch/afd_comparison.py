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

"""Launch one equal-four-GPU arm for the AFD versus PD comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from afd_launcher import preflight as preflight_afd_stack
from afd_launcher import wait_for_ffn_daemon
from dynamo.vllm.afd_supervisor import Supervisor

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V2-Lite"
DEFAULT_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
REQUIRED_VLLM = "0.26.0"
EXPECTED_DEVICES = {"0", "1", "2", "3"}
KV_TRANSFER_CONFIG = json.dumps(
    {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    separators=(",", ":"),
)


@dataclass(frozen=True)
class Settings:
    arm: str
    model: str
    revision: str
    log_dir: Path
    namespace: str
    port_base: int
    enable_dbo: bool = False
    pd_prefill_workers: int = 2
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 64


@dataclass
class ProcessSpec:
    name: str
    role: str
    devices: tuple[str, ...]
    command: list[str]
    environment: dict[str, str]

    def manifest(self) -> dict[str, object]:
        env_keys = (
            "CUDA_VISIBLE_DEVICES",
            "DYN_SYSTEM_PORT",
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "VLLM_NIXL_SIDE_CHANNEL_PORT",
            "UCX_TLS",
            "DYN_NAMESPACE",
            "DYN_DISCOVERY_BACKEND",
            "DYN_REQUEST_PLANE",
            "DYN_EVENT_PLANE",
            "DYN_FILE_KV",
            "VLLM_PLUGINS",
            "MODEL",
            "MODEL_REVISION",
            "AFD_LOG_DIR",
            "AFD_ATTENTION_DEVICES",
            "AFD_FFN_DEVICES",
            "AFD_ATTENTION_DP",
            "AFD_ATTENTION_TP",
            "AFD_FFN_DP",
            "AFD_FFN_TP",
            "AFD_PORT",
            "AFD_FFN_HTTP_PORT",
            "AFD_ENABLE_DBO",
            "MAX_MODEL_LEN",
            "MAX_CONCURRENT_SEQS",
            "MAX_BATCHED_TOKENS",
        )
        return {
            "name": self.name,
            "role": self.role,
            "devices": list(self.devices),
            "command": self.command,
            "environment": {
                key: self.environment[key]
                for key in env_keys
                if key in self.environment
            },
        }


@dataclass
class ComparisonPlan:
    settings: Settings
    processes: list[ProcessSpec]
    ports: dict[str, int]
    readiness_urls: list[str]

    def manifest(self) -> dict[str, object]:
        return {
            "arm": self.settings.arm,
            "model": self.settings.model,
            "model_revision": self.settings.revision,
            "dtype": "bfloat16",
            "enable_dbo": self.settings.enable_dbo,
            "pd_prefill_workers": self.settings.pd_prefill_workers
            if self.settings.arm == "pd"
            else None,
            "dynamo_namespace": self.settings.namespace,
            "ports": self.ports,
            "readiness_urls": self.readiness_urls,
            "processes": [process.manifest() for process in self.processes],
        }


def common_engine_args(settings: Settings) -> list[str]:
    args = [
        "--revision",
        settings.revision,
        "--tokenizer-revision",
        settings.revision,
        "--dtype",
        "bfloat16",
        "--enable-expert-parallel",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(settings.max_model_len),
        "--max-num-seqs",
        str(settings.max_num_seqs),
        "--max-num-batched-tokens",
        str(settings.max_num_batched_tokens),
        "--enforce-eager",
        "--trust-remote-code",
    ]
    return args


def afd_config(role: str, port: int, attention_ranks: int, ffn_ranks: int) -> str:
    return json.dumps(
        {
            "afd": {
                "role": role,
                "connector": "P2pNcclAFDConnector",
                "host": "127.0.0.1",
                "port": port,
                "num_attention_ranks": attention_ranks,
                "num_ffn_ranks": ffn_ranks,
            }
        },
        separators=(",", ":"),
    )


def runtime_environment(
    settings: Settings, source_env: Mapping[str, str]
) -> dict[str, str]:
    environment = dict(source_env)
    for key in (
        "DYN_NAMESPACE_PREFIX",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_ENDPOINT",
        "DYN_KV_STATE_ENDPOINT",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "DYN_NAMESPACE": settings.namespace,
            "DYN_DISCOVERY_BACKEND": "file",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_FILE_KV": str(settings.log_dir.resolve() / "discovery"),
            "DYN_HTTP_PORT": str(settings.port_base),
            "DYN_HTTP_HOST": "127.0.0.1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_PLUGINS": "afd" if "afd" in settings.arm else "",
        }
    )
    return environment


def dynamo_worker(
    settings: Settings,
    base_env: Mapping[str, str],
    *,
    name: str,
    role: str,
    devices: tuple[str, ...],
    system_port: int,
    dp: int,
    tp: int,
    extra_args: list[str] | None = None,
    nixl_port: int | None = None,
) -> ProcessSpec:
    environment = dict(base_env)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
            "DYN_SYSTEM_PORT": str(system_port),
        }
    )
    if nixl_port is not None:
        environment["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
        environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_port)
        environment["UCX_TLS"] = "cuda_ipc,cuda_copy,tcp"
    command = [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        settings.model,
        "--data-parallel-size",
        str(dp),
        "--tensor-parallel-size",
        str(tp),
        *common_engine_args(settings),
        *(extra_args or []),
    ]
    return ProcessSpec(name, role, devices, command, environment)


def event_config(port: int) -> str:
    return json.dumps(
        {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{port}",
            "enable_kv_cache_events": True,
        },
        separators=(",", ":"),
    )


def build_plan(
    settings: Settings, source_env: Mapping[str, str] | None = None
) -> ComparisonPlan:
    if settings.arm not in {"agg", "pd", "afd", "pd-afd"}:
        raise ValueError(f"unknown comparison arm: {settings.arm}")
    if not 1024 <= settings.port_base <= 65490:
        raise ValueError("port_base must be between 1024 and 65490")
    if not settings.namespace.strip() or settings.namespace == "dynamo":
        raise ValueError("comparison namespace must be non-empty and isolated")
    if settings.enable_dbo:
        raise ValueError(
            "matched comparison DBO is not qualified: native workers require an "
            "explicit supported all2all backend; keep DBO disabled for all arms"
        )
    if settings.pd_prefill_workers not in {1, 2}:
        raise ValueError("pd_prefill_workers must be 1 or 2")

    base_env = runtime_environment(
        settings, os.environ if source_env is None else source_env
    )
    frontend = ProcessSpec(
        "frontend",
        "frontend",
        (),
        [sys.executable, "-m", "dynamo.frontend"],
        dict(base_env),
    )
    ports = {"frontend": settings.port_base}
    readiness = [f"http://127.0.0.1:{settings.port_base}/v1/models"]
    processes = [frontend]

    if settings.arm == "agg":
        system_port = settings.port_base + 10
        processes.append(
            dynamo_worker(
                settings,
                base_env,
                name="aggregated",
                role="aggregated",
                devices=("0", "1", "2", "3"),
                system_port=system_port,
                dp=4,
                tp=1,
            )
        )
        ports["aggregated_system"] = system_port
        readiness.append(f"http://127.0.0.1:{system_port}/health")

    elif settings.arm == "pd":
        for index in range(settings.pd_prefill_workers):
            prefill_system = settings.port_base + 10 + index
            prefill_nixl = settings.port_base + 20 + index
            event_port = settings.port_base + 30 + index
            processes.append(
                dynamo_worker(
                    settings,
                    base_env,
                    name=f"prefill-{index}",
                    role="prefill",
                    devices=(str(index),),
                    system_port=prefill_system,
                    dp=1,
                    tp=1,
                    nixl_port=prefill_nixl,
                    extra_args=[
                        "--disaggregation-mode",
                        "prefill",
                        "--kv-transfer-config",
                        KV_TRANSFER_CONFIG,
                        "--kv-events-config",
                        event_config(event_port),
                    ],
                )
            )
            ports.update(
                {
                    f"prefill_{index}_system": prefill_system,
                    f"prefill_{index}_nixl": prefill_nixl,
                    f"prefill_{index}_events": event_port,
                }
            )
            readiness.append(f"http://127.0.0.1:{prefill_system}/health")
        for index, device in enumerate(range(settings.pd_prefill_workers, 4)):
            system_port = settings.port_base + 10 + device
            nixl_port = settings.port_base + 20 + device
            processes.append(
                dynamo_worker(
                    settings,
                    base_env,
                    name=f"decode-{index}",
                    role="decode",
                    devices=(str(device),),
                    system_port=system_port,
                    dp=1,
                    tp=1,
                    nixl_port=nixl_port,
                    extra_args=[
                        "--disaggregation-mode",
                        "decode",
                        "--kv-transfer-config",
                        KV_TRANSFER_CONFIG,
                    ],
                )
            )
            ports[f"decode_{index}_system"] = system_port
            ports[f"decode_{index}_nixl"] = nixl_port
            readiness.append(f"http://127.0.0.1:{system_port}/health")

    elif settings.arm == "afd":
        afd_port = settings.port_base + 40
        attention_system = settings.port_base + 10
        processes.append(
            dynamo_worker(
                settings,
                base_env,
                name="attention",
                role="attention",
                devices=("0", "1"),
                system_port=attention_system,
                dp=1,
                tp=2,
                extra_args=[
                    "--additional-config",
                    afd_config("attention", afd_port, 2, 2),
                    "--worker-cls",
                    "afd_plugin.v1.worker.AFDAttentionWorker",
                ],
            )
        )
        ffn_env = dict(base_env)
        ffn_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        ffn_command = [
            "vllm",
            "serve",
            settings.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.port_base + 41),
            "--data-parallel-size",
            "1",
            "--tensor-parallel-size",
            "2",
            "--additional-config",
            afd_config("ffn", afd_port, 2, 2),
            "--worker-cls",
            "afd_plugin.v1.worker.AFDFFNWorker",
            *common_engine_args(settings),
        ]
        processes.append(ProcessSpec("ffn", "ffn", ("2", "3"), ffn_command, ffn_env))
        ports.update(
            {
                "attention_system": attention_system,
                "afd_rendezvous": afd_port,
                "ffn_lifecycle": settings.port_base + 41,
            }
        )
        readiness.append(f"http://127.0.0.1:{attention_system}/health")

    else:
        afd_port = settings.port_base + 40
        for index, device in enumerate(("0", "1")):
            system_port = settings.port_base + 10 + index
            nixl_port = settings.port_base + 20 + index
            event_port = settings.port_base + 30 + index
            processes.append(
                dynamo_worker(
                    settings,
                    base_env,
                    name=f"prefill-{index}",
                    role="prefill",
                    devices=(device,),
                    system_port=system_port,
                    dp=1,
                    tp=1,
                    nixl_port=nixl_port,
                    extra_args=[
                        "--disaggregation-mode",
                        "prefill",
                        "--kv-transfer-config",
                        KV_TRANSFER_CONFIG,
                        "--kv-events-config",
                        event_config(event_port),
                    ],
                )
            )
            ports.update(
                {
                    f"prefill_{index}_system": system_port,
                    f"prefill_{index}_nixl": nixl_port,
                    f"prefill_{index}_events": event_port,
                }
            )
            readiness.append(f"http://127.0.0.1:{system_port}/health")

        decode_system = settings.port_base + 12
        decode_nixl = settings.port_base + 22
        processes.append(
            dynamo_worker(
                settings,
                base_env,
                name="decode-attention",
                role="decode-attention",
                devices=("2",),
                system_port=decode_system,
                dp=1,
                tp=1,
                nixl_port=decode_nixl,
                extra_args=[
                    "--disaggregation-mode",
                    "decode",
                    "--kv-transfer-config",
                    KV_TRANSFER_CONFIG,
                    "--additional-config",
                    afd_config("attention", afd_port, 1, 1),
                    "--worker-cls",
                    "afd_plugin.v1.worker.AFDAttentionWorker",
                ],
            )
        )
        ffn_env = dict(base_env)
        ffn_env["CUDA_VISIBLE_DEVICES"] = "3"
        ffn_command = [
            "vllm",
            "serve",
            settings.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.port_base + 41),
            "--data-parallel-size",
            "1",
            "--tensor-parallel-size",
            "1",
            "--additional-config",
            afd_config("ffn", afd_port, 1, 1),
            "--worker-cls",
            "afd_plugin.v1.worker.AFDFFNWorker",
            *common_engine_args(settings),
        ]
        processes.append(ProcessSpec("ffn", "ffn", ("3",), ffn_command, ffn_env))
        ports.update(
            {
                "decode_attention_system": decode_system,
                "decode_attention_nixl": decode_nixl,
                "afd_rendezvous": afd_port,
                "ffn_lifecycle": settings.port_base + 41,
            }
        )
        readiness.append(f"http://127.0.0.1:{decode_system}/health")

    validate_plan(processes, ports)
    return ComparisonPlan(settings, processes, ports, readiness)


def validate_plan(processes: list[ProcessSpec], ports: Mapping[str, int]) -> None:
    devices = [device for process in processes for device in process.devices]
    if set(devices) != EXPECTED_DEVICES or len(devices) != len(EXPECTED_DEVICES):
        raise ValueError(
            f"comparison arms must allocate devices 0,1,2,3 exactly once; got {devices}"
        )
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"comparison arm contains colliding ports: {ports}")
    if any(port > 65535 for port in ports.values()):
        raise ValueError("comparison arm contains a port greater than 65535")


def preflight(arm: str) -> dict[str, object]:
    if "afd" in arm:
        return preflight_afd_stack()
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("required package is not installed: vllm") from error
    if vllm_version != REQUIRED_VLLM:
        raise RuntimeError(f"vLLM {REQUIRED_VLLM} is required; found {vllm_version}")
    return {"vllm": vllm_version}


def require_free_ports(ports: Mapping[str, int]) -> None:
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"{name}={port} is unavailable") from error


def wait_until_ready(
    supervisor: Supervisor,
    urls: list[str],
    model_url: str,
    expected_model: str,
    timeout: int,
) -> None:
    pending = set(urls)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        if supervisor.signal_number is not None:
            raise KeyboardInterrupt
        if exited := supervisor.exited_child():
            raise RuntimeError(
                f"{exited.name} exited with {exited.process.returncode}; "
                f"see {exited.log_path}"
            )
        for url in tuple(pending):
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if not 200 <= response.status < 300:
                        continue
                    if url == model_url:
                        payload = json.load(response)
                        model_ids = {
                            item.get("id")
                            for item in payload.get("data", [])
                            if isinstance(item, dict)
                        }
                        if expected_model not in model_ids:
                            continue
                    pending.remove(url)
            except (OSError, ValueError, urllib.error.URLError):
                pass
        if pending:
            time.sleep(1)
    if pending:
        raise TimeoutError(f"readiness timed out with pending endpoints: {pending}")


def run_plan(plan: ComparisonPlan, timeout: int) -> int:
    versions = preflight(plan.settings.arm)
    require_free_ports(plan.ports)
    supervisor = Supervisor(plan.settings.log_dir)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, supervisor.handle_signal)
    manifest = plan.manifest() | {"versions": versions}
    (plan.settings.log_dir / "comparison-plan.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    try:
        for process in plan.processes:
            supervisor.start(process.name, process.command, process.environment)
        model_url = plan.readiness_urls[0]
        wait_until_ready(
            supervisor,
            plan.readiness_urls,
            model_url,
            plan.settings.model,
            timeout,
        )
        if plan.settings.arm in {"afd", "pd-afd"}:
            wait_for_ffn_daemon(supervisor, timeout=min(timeout, 30))
        print(
            f"ready arm={plan.settings.arm} model={plan.settings.model} "
            f"logs={plan.settings.log_dir}",
            flush=True,
        )
        return supervisor.wait()
    except KeyboardInterrupt:
        return 128 + (supervisor.signal_number or signal.SIGINT)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"comparison launch failed: {error}", file=sys.stderr)
        return 1
    finally:
        supervisor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("agg", "pd", "afd", "pd-afd"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--namespace")
    parser.add_argument("--port-base", type=int, default=18000)
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help="Unsupported until a matched native backend is qualified",
    )
    parser.add_argument(
        "--pd-prefill-workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="PD prefill count; 2 matches PD+AFD's prefill footprint",
    )
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex
    log_dir = args.log_dir or Path(f"/tmp/dynamo-afd-{args.arm}-{run_id}")
    settings = Settings(
        arm=args.arm,
        model=args.model,
        revision=args.revision,
        log_dir=log_dir,
        namespace=args.namespace or f"dynamo-afd-{args.arm}-{run_id}",
        port_base=args.port_base,
        enable_dbo=args.enable_dbo,
        pd_prefill_workers=args.pd_prefill_workers,
    )
    plan = build_plan(settings)
    if args.print_plan:
        print(json.dumps(plan.manifest(), indent=2))
        return 0
    return run_plan(plan, args.ready_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
