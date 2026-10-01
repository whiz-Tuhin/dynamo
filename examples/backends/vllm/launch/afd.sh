#!/bin/bash
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

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# AFD's native FFN worker has a separate process tree, so the standard
# `kill 0` EXIT trap could also signal this script's caller.
# The Python supervisor detects exits and cleans up one owned session per role.
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
export AFD_VLLM_GPU_MEM_ARGS
AFD_VLLM_GPU_MEM_ARGS="$(build_vllm_gpu_mem_args)"
exec python3 "$SCRIPT_DIR/afd_launcher.py" "$@"
