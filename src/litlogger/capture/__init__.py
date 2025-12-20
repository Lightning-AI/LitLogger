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
"""Output capture utilities for litlogger.

This package provides platform-specific output capture implementations:
- POSIX systems (Linux, macOS): Uses pseudo-terminals (PTY) for full terminal emulation
- Windows: Uses subprocess pipes (PTY is not available on Windows)
"""

import sys


def rerun_and_record(terminal_logs_path: str) -> None:
    """Re-exec the script and record output to a log file.

    Uses the appropriate implementation based on the platform:
    - POSIX: PTY-based capture for full terminal emulation
    - Windows: Pipe-based capture

    Args:
        terminal_logs_path: Path to the file where terminal logs will be recorded.
    """
    if sys.platform == "win32":
        # Import lazily to avoid loading POSIX-only modules on Windows
        from litlogger.capture.windows import rerun_and_record_windows

        rerun_and_record_windows(terminal_logs_path)
    else:
        # Import lazily to avoid loading Windows-only code on POSIX
        from litlogger.capture.posix import rerun_in_pty_and_record

        rerun_in_pty_and_record(terminal_logs_path)


__all__ = ["rerun_and_record"]
