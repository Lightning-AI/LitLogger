# Copyright The Lightning AI team.
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
"""Utility functions for litlogger."""

import os
import select
import subprocess
import sys

from rich.text import Text


def rerun_in_pty_and_record(terminal_logs_path: str) -> None:
    """Re-exec under a pseudo-terminal, mirror output to stdout, and record it to a log file.

    Args:
        terminal_logs_path: Path to the file where terminal logs will be recorded.
    """
    import pty  # this needs to be lazily imported to avoid loading it on Windows

    command = [sys.executable, *sys.argv]
    env = os.environ.copy()
    env["_IN_PTY_RECORDER"] = "1"

    with open(terminal_logs_path, "wb") as log_file:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(command, env=env, stdout=slave_fd, stderr=slave_fd, stdin=None)
        os.close(slave_fd)
        try:
            while True:
                try:
                    ready_fds, _, _ = select.select([master_fd], [], [], 0.1)
                    if not ready_fds:
                        if process.poll() is not None:
                            break
                        continue
                    data = os.read(master_fd, 1024)
                except OSError:
                    break
                if not data:
                    break
                # Write raw data with ANSI codes to stdout for colored terminal output
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
                # Strip ANSI escape codes from log file since they can't be displyed anyways
                data_str = data.decode("utf-8", errors="replace")
                data_no_ansi = Text.from_ansi(data_str).plain
                log_file.write(data_no_ansi.encode("utf-8"))
        finally:
            os.close(master_fd)
            if process.poll() is None:
                process.terminate()
            process.wait()
