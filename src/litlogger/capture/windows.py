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
"""Windows-compatible output capture utility for litlogger.

This module provides a Windows alternative to the PTY-based output capture.
It uses subprocess pipes instead of pseudo-terminals.
"""

import os
import subprocess
import sys
import threading
from typing import BinaryIO

from rich.text import Text


def _stream_reader(stream: BinaryIO, log_file: BinaryIO, lock: threading.Lock) -> None:
    """Read from a stream and write to both stdout and log file.

    Args:
        stream: The subprocess stream to read from.
        log_file: The file to write stripped output to.
        lock: Thread lock for synchronized file writes.
    """
    try:
        while True:
            data = stream.read(1024)
            if not data:
                break

            # Write raw data to stdout for terminal output
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

            # Strip ANSI escape codes from log file
            data_str = data.decode("utf-8", errors="replace")
            data_no_ansi = Text.from_ansi(data_str).plain

            with lock:
                log_file.write(data_no_ansi.encode("utf-8"))
    except Exception:
        pass  # Ignore errors during stream reading


def rerun_and_record_windows(terminal_logs_path: str) -> None:
    """Re-exec the script and record output to a log file (Windows version).

    This is a Windows-compatible alternative to rerun_in_pty_and_record.
    It uses subprocess pipes instead of pseudo-terminals.

    Note: While this would theoretically be platform agnostic,
    some programs may detect they're not running in a TTY and disable
    colored output. We set FORCE_COLOR and PYTHONUNBUFFERED to encourage
    color output, but this is not guaranteed for all programs. So we only use
    this for windows to get as close as possible to the PTY-based approach.

    Args:
        terminal_logs_path: Path to the file where terminal logs will be recorded.
    """
    command = [sys.executable, *sys.argv]
    env = os.environ.copy()
    env["_IN_PTY_RECORDER"] = "1"
    # Encourage colored output
    env["FORCE_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    with open(terminal_logs_path, "wb") as log_file:
        # Use pipes for stdout and stderr
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            stdin=None,
        )

        lock = threading.Lock()

        # Read and process output
        if process.stdout:
            _stream_reader(process.stdout, log_file, lock)

        process.wait()
