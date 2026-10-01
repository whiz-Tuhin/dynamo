<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental Attention-FFN disaggregation

This prototype connects a Dynamo request-facing Attention worker to an external vLLM FFN worker through the public `P2pNcclAFDConnector`. The Attention role owns request scheduling, KV cache, residual addition, sampling, and the Dynamo endpoint. The FFN role is a connector-driven sidecar and does not accept Dynamo requests.

> [!WARNING]
> This path is experimental. It has source-level and CPU process-lifecycle checks, but no GPU end-to-end or performance result is included. Do not infer an AFD speedup from a successful launch.

## Compatibility

Use these public revisions together:

- Dynamo: `f5802d355b59b5006c67a16c3ed1615cfcab5686`
- [vLLM AFD plugin](https://github.com/vllm-project/afd-plugin): `9cae2d2ddaae7e1752fc12ead577d59540b17f27`
- vLLM: `0.26.0`
- Model and tokenizer: `deepseek-ai/DeepSeek-V2-Lite@604d5664dddd88a0433dbae533b7fe9472482de0`

The Dynamo revision is the last public commit before its vLLM runtime advanced to 0.27.1. The selected plugin revision declares vLLM 0.26.0. Install that plugin checkout into the same environment as Dynamo:

```bash
uv pip install -e '/path/to/afd-plugin[vllm]'
```

The launcher verifies vLLM 0.26.0, the installed `afd` entry point and compatibility patches, role topology, device allocation, and local ports before starting processes. It forces the AFD plugin, vLLM V1 runner, BF16, static expert parallelism, and the pinned model/tokenizer revision.

## Run the four-GPU smoke

The initial layout is 2 Attention GPUs and 2 FFN GPUs, with DP1TP2 for each role:

```bash
export AFD_LOG_DIR=/tmp/afd-dynamo-smoke
export DYN_HTTP_PORT=18000
export DYN_SYSTEM_PORT=18081
export AFD_PORT=16269
export AFD_FFN_HTTP_PORT=18001
examples/backends/vllm/launch/afd.sh
```

Override `AFD_ATTENTION_DEVICES`, `AFD_FFN_DEVICES`, and the corresponding `AFD_*_DP` and `AFD_*_TP` variables together. Device lists must be unique and nonoverlapping, their lengths must equal DP times TP, and this connector requires at least as many Attention ranks as FFN ranks.

The launcher creates a unique `DYN_NAMESPACE` unless one is supplied. Empty namespaces and the global namespace `dynamo` are rejected. It reports readiness only when that namespace's frontend lists the expected model and the FFN engine acknowledges that its connector loop started. Both AFD roles receive explicit worker classes; a plugin import failure cannot silently substitute a native vLLM worker. It writes `frontend.log`, `attention.log`, `ffn.log`, and `launch.json` under `AFD_LOG_DIR`. The manifest contains the installed package identity, imported plugin path, model revision, resolved commands, device mapping, namespace, and ports.

For the single-node smoke, discovery defaults to a local file store under the log directory, requests use TCP, and events use ZMQ. This avoids requiring a separate etcd or NATS deployment. Explicit `DYN_DISCOVERY_BACKEND`, `DYN_REQUEST_PLANE`, `DYN_EVENT_PLANE` and `DYN_FILE_KV` settings are honored and recorded; supply their dependencies when overriding. Inherited namespace prefixes, worker suffixes and explicit endpoint identities are cleared so they cannot redirect this standalone launch into another deployment.

DBO is disabled for the first correctness smoke. Enable it as a separate experiment with:

```bash
export AFD_ENABLE_DBO=1
examples/backends/vllm/launch/afd.sh
```

The synchronous connector exchanges activations at every remotely split layer. Measure that communication and both roles' idle time before attributing a gain or loss. Compare AFD against tuned aggregated, prefill/decode-disaggregated, and combined PD+AFD arms on the same GPU budget and request trace. Preserve errors and unfinished requests alongside latency and throughput.

## Failure handling

Frontend, Attention, and FFN run in separate owned process groups with separate logs. The supervisor fails when any role exits, terminates every recorded role group, and does not signal the caller's process group. The FFN CLI port is a lifecycle input; do not use it as a request or readiness endpoint.

After readiness, verify the logs show the AFD Attention and FFN worker classes and the AFD DeepSeek model wrapper. Then run deterministic reference/candidate/reference requests before collecting performance. A successful HTTP response alone does not prove correct AFD wiring.
