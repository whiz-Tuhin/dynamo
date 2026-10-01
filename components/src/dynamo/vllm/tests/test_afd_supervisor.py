# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for AFD role process-group ownership."""

import os
import subprocess
import sys
import time

import pytest
from dynamo.vllm.afd_supervisor import Supervisor, process_group_exists

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_cleanup_kills_orphaned_role_group_but_not_outside_process(tmp_path):
    child_pid_path = tmp_path / "grandchild.pid"
    role_script = """
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w", encoding="utf-8") as output:
    output.write(str(child.pid))
"""
    supervisor = Supervisor(tmp_path / "logs")
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    ) as outside:
        try:
            supervisor.start(
                "role",
                [sys.executable, "-c", role_script, str(child_pid_path)],
                os.environ.copy(),
            )
            deadline = time.monotonic() + 5
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child_pid_path.exists()
            role = supervisor.children[0]
            role.process.wait(timeout=5)
            assert process_group_exists(role.process_group)

            supervisor.close()

            assert not process_group_exists(role.process_group)
            assert outside.poll() is None
        finally:
            if any(
                process_group_exists(child.process_group)
                for child in supervisor.children
            ):
                supervisor.close()
            if outside.poll() is None:
                outside.terminate()
