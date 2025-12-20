# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for litlogger.capture.windows module.

While this should work on every platform,
we only test it on Windows as that's the only platform we use it on.
"""

import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest
from litlogger.capture.windows import rerun_and_record_windows

if sys.platform != "win32":
    pytest.skip("Windows only test", allow_module_level=True)


class TestRerunAndRecordWindows:
    """Test the rerun_and_record_windows function."""

    @patch("litlogger.capture.windows.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_spawns_subprocess(self, mock_file, mock_popen):
        """Test that subprocess is spawned with correct parameters."""
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.read.return_value = b""
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with (
            patch.object(sys, "executable", "python.exe"),
            patch.object(sys, "argv", ["script.py", "arg1", "arg2"]),
        ):
            rerun_and_record_windows("C:\\temp\\logs.txt")

        # Verify subprocess was spawned correctly
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args

        # Check command
        assert call_args[0][0] == ["python.exe", "script.py", "arg1", "arg2"]

        # Check that environment has _IN_PTY_RECORDER set
        assert call_args[1]["env"]["_IN_PTY_RECORDER"] == "1"

        # Check that FORCE_COLOR is set to encourage color output
        assert call_args[1]["env"]["FORCE_COLOR"] == "1"

        # Check that PYTHONUNBUFFERED is set
        assert call_args[1]["env"]["PYTHONUNBUFFERED"] == "1"

        # Check pipe setup
        assert call_args[1]["stdout"] is not None

    @patch("litlogger.capture.windows.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("litlogger.capture.windows.sys.stdout")
    def test_rerun_writes_output_to_stdout(self, mock_stdout, mock_file, mock_popen):
        """Test that output is written to stdout."""
        mock_process = MagicMock()

        # Create a mock stdout that returns data then empty bytes
        mock_pipe = MagicMock()
        mock_pipe.read.side_effect = [b"test output\n", b""]
        mock_process.stdout = mock_pipe
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        mock_stdout.buffer = MagicMock()

        rerun_and_record_windows("/tmp/logs.txt")

        # Verify data was written to stdout
        mock_stdout.buffer.write.assert_called()
        assert b"test output\n" in [c[0][0] for c in mock_stdout.buffer.write.call_args_list]

    @patch("litlogger.capture.windows.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("litlogger.capture.windows.sys.stdout")
    def test_rerun_writes_to_log_file(self, mock_stdout, mock_file_open, mock_popen):
        """Test that output is written to log file without ANSI codes."""
        mock_process = MagicMock()

        # Create a mock stdout that returns data with ANSI codes then empty bytes
        mock_pipe = MagicMock()
        mock_pipe.read.side_effect = [b"\x1b[31mRed text\x1b[0m\n", b""]
        mock_process.stdout = mock_pipe
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        mock_stdout.buffer = MagicMock()

        rerun_and_record_windows("/tmp/logs.txt")

        # Verify file was opened in write-binary mode
        mock_file_open.assert_called_with("/tmp/logs.txt", "wb")

        # Get the mock file handle
        mock_file_handle = mock_file_open()

        # Verify write was called on the file
        mock_file_handle.write.assert_called()

    @patch("litlogger.capture.windows.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_waits_for_process_completion(self, mock_file, mock_popen):
        """Test that rerun waits for subprocess to complete."""
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.read.return_value = b""
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        rerun_and_record_windows("/tmp/logs.txt")

        # Verify wait was called
        mock_process.wait.assert_called_once()

    @patch("litlogger.capture.windows.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("litlogger.capture.windows.sys.stdout")
    def test_rerun_handles_empty_data(self, mock_stdout, mock_file, mock_popen):
        """Test that empty data breaks the loop."""
        mock_process = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.read.return_value = b""  # Empty data immediately
        mock_process.stdout = mock_pipe
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        mock_stdout.buffer = MagicMock()

        rerun_and_record_windows("/tmp/logs.txt")

        # Loop should break and cleanup should happen
        mock_process.wait.assert_called_once()
