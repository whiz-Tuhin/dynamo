# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process supervision for an AFD deployment with an external FFN role."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass
class Child:
    name: str
    process: subprocess.Popen[bytes]
    process_group: int
    log: IO[bytes]
    log_path: Path


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except (PermissionError, ProcessLookupError):
        # These groups are created by this process under our UID. macOS
        # returns EPERM once a signalled group contains only an orphan zombie;
        # it has no live process left to clean up.
        return False
    return True


class Supervisor:
    """Run each role in its own process group and clean up every recorded group."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.children: list[Child] = []
        self.signal_number: int | None = None
        self.exit_stack = ExitStack()

    def handle_signal(self, signum: int, _frame: object) -> None:
        self.signal_number = signum

    def start(self, name: str, command: list[str], env: dict[str, str]) -> None:
        log_path = self.log_dir / f"{name}.log"
        log = self.exit_stack.enter_context(log_path.open("wb"))
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # start_new_session=True makes the child the new session and process
        # group leader, so its PID is the PGID even if the leader exits here.
        self.children.append(Child(name, process, process.pid, log, log_path))
        print(f"started {name} pid={process.pid} log={log_path}", flush=True)

    def exited_child(self) -> Child | None:
        return next(
            (child for child in self.children if child.process.poll() is not None),
            None,
        )

    def wait(self) -> int:
        while self.signal_number is None:
            exited = self.exited_child()
            if exited is not None:
                code = exited.process.returncode
                print(
                    f"{exited.name} exited with {code}; see {exited.log_path}",
                    file=sys.stderr,
                )
                return code if code else 1
            time.sleep(0.5)
        return 128 + self.signal_number

    def _signal_groups(self, signum: int) -> None:
        for child in reversed(self.children):
            try:
                os.killpg(child.process_group, signum)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        """Stop whole role groups, including descendants of exited leaders."""
        self._signal_groups(signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for child in self.children:
                child.process.poll()
            if not any(
                process_group_exists(child.process_group) for child in self.children
            ):
                break
            time.sleep(0.1)

        survivors = [
            child
            for child in self.children
            if process_group_exists(child.process_group)
        ]
        for child in survivors:
            try:
                os.killpg(child.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in self.children:
            try:
                child.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        kill_deadline = time.monotonic() + 5
        while time.monotonic() < kill_deadline and any(
            process_group_exists(child.process_group) for child in self.children
        ):
            time.sleep(0.1)
        remaining = [
            child.process_group
            for child in self.children
            if process_group_exists(child.process_group)
        ]
        self.exit_stack.close()
        if remaining:
            raise RuntimeError(f"AFD process groups survived cleanup: {remaining}")
