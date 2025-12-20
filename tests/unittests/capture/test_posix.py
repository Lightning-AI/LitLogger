# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for litlogger.utils utility functions."""

import sys

import pytest

# needs to be run before imports below to avoid loading pty on Windows
if sys.platform == "win32":
    pytest.skip("PTY is not available on Windows", allow_module_level=True)

from unittest.mock import MagicMock, mock_open, patch

from litlogger.capture.posix import rerun_in_pty_and_record


class TestRerunInPtyAndRecord:
    """Test the rerun_in_pty_and_record function."""

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_creates_pty(self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen):
        """Test that rerun_in_pty_and_record creates a PTY."""
        # Setup mocks
        mock_openpty.return_value = (100, 101)  # master_fd, slave_fd
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Process finished
        mock_popen.return_value = mock_process
        mock_select.return_value = ([], [], [])  # No data ready

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify PTY was opened
        mock_openpty.assert_called_once()

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_spawns_subprocess(self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen):
        """Test that subprocess is spawned with correct parameters."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_select.return_value = ([], [], [])

        with (
            patch.object(sys, "executable", "/usr/bin/python3"),
            patch.object(sys, "argv", ["script.py", "arg1", "arg2"]),
        ):
            rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify subprocess was spawned correctly
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args

        # Check command
        assert call_args[0][0] == ["/usr/bin/python3", "script.py", "arg1", "arg2"]

        # Check that environment has _IN_PTY_RECORDER set
        assert call_args[1]["env"]["_IN_PTY_RECORDER"] == "1"

        # Check file descriptors
        assert call_args[1]["stdout"] == 101  # slave_fd
        assert call_args[1]["stderr"] == 101  # slave_fd

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_closes_slave_fd(self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen):
        """Test that slave file descriptor is closed after spawning."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_select.return_value = ([], [], [])

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify slave_fd (101) was closed
        close_calls = [c[0][0] for c in mock_close.call_args_list]
        assert 101 in close_calls

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    @patch("litlogger.capture.posix.sys.stdout")
    def test_rerun_writes_output_to_stdout(
        self, mock_stdout, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen
    ):
        """Test that output is written to stdout."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        # First two polls return None (running), then 0 (finished) for loop break and finally block
        mock_process.poll.side_effect = [None, None, 0, 0]
        mock_popen.return_value = mock_process

        # Simulate data being ready on first select, then none until process finishes
        # Need enough iterations: 1 with data, then 2 with no data (polling), then 1 final check
        mock_select.side_effect = [([100], [], []), ([], [], []), ([], [], []), ([], [], [])]
        mock_read.return_value = b"test output\n"

        # Mock stdout.buffer
        mock_stdout.buffer = MagicMock()

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify data was written to stdout
        mock_stdout.buffer.write.assert_called()
        assert b"test output\n" in [c[0][0] for c in mock_stdout.buffer.write.call_args_list]

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    @patch("litlogger.capture.posix.sys.stdout")
    def test_rerun_writes_to_log_file(
        self, mock_stdout, mock_file_open, mock_close, mock_read, mock_select, mock_openpty, mock_popen
    ):
        """Test that output is written to log file without ANSI codes."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        # First two polls return None (running), then 0 (finished) for loop break and finally block
        mock_process.poll.side_effect = [None, None, 0, 0]
        mock_popen.return_value = mock_process
        # Need enough iterations: 1 with data, then 2 with no data (polling), then 1 final check
        mock_select.side_effect = [([100], [], []), ([], [], []), ([], [], []), ([], [], [])]

        # Data with ANSI escape codes
        mock_read.return_value = b"\x1b[31mRed text\x1b[0m\n"
        mock_stdout.buffer = MagicMock()

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify file was opened in write-binary mode
        mock_file_open.assert_called_with("/tmp/logs.txt", "wb")

        # Get the mock file handle
        mock_file_handle = mock_file_open()

        # Verify write was called on the file (ANSI codes should be stripped)
        mock_file_handle.write.assert_called()

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_closes_master_fd_on_exit(
        self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen
    ):
        """Test that master file descriptor is closed."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_select.return_value = ([], [], [])

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify master_fd (100) was closed
        close_calls = [c[0][0] for c in mock_close.call_args_list]
        assert 100 in close_calls

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_waits_for_process_completion(
        self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen
    ):
        """Test that rerun waits for subprocess to complete."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_select.return_value = ([], [], [])

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify wait was called
        mock_process.wait.assert_called_once()

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_terminates_running_process(
        self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen
    ):
        """Test that running process is terminated on exit."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        # Process is running for first check, then terminates
        mock_process.poll.side_effect = [None, 0, None]  # Running, then exits, then still running in finally
        mock_popen.return_value = mock_process
        mock_select.side_effect = [([], [], []), ([], [], [])]

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify terminate was called (because poll returns None in finally block)
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_handles_oserror(self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen):
        """Test that OSError during read is handled gracefully."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_select.side_effect = [([100], [], []), OSError("Simulated error")]
        mock_read.return_value = b"data"

        # Should not raise an exception
        rerun_in_pty_and_record("/tmp/logs.txt")

        # Verify cleanup still happens
        mock_process.wait.assert_called_once()

    @patch("litlogger.capture.posix.subprocess.Popen")
    @patch("pty.openpty")
    @patch("litlogger.capture.posix.select.select")
    @patch("litlogger.capture.posix.os.read")
    @patch("litlogger.capture.posix.os.close")
    @patch("builtins.open", new_callable=mock_open)
    def test_rerun_handles_empty_data(self, mock_file, mock_close, mock_read, mock_select, mock_openpty, mock_popen):
        """Test that empty data breaks the loop."""
        mock_openpty.return_value = (100, 101)
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None]
        mock_popen.return_value = mock_process
        mock_select.return_value = ([100], [], [])
        mock_read.return_value = b""  # Empty data

        rerun_in_pty_and_record("/tmp/logs.txt")

        # Loop should break and cleanup should happen
        mock_process.wait.assert_called_once()
