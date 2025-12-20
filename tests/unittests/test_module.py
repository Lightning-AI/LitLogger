# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for litlogger._module global state management."""

from unittest.mock import MagicMock

import litlogger
import pytest
from litlogger._module import set_global, unset_globals


class TestSetGlobal:
    """Test the set_global function."""

    def test_set_experiment(self):
        """Test setting only the experiment."""
        mock_experiment = MagicMock()
        original_log = litlogger.log

        set_global(experiment=mock_experiment)

        assert litlogger.experiment is mock_experiment
        # Other globals should remain unchanged
        assert litlogger.log is original_log

    def test_set_log(self):
        """Test setting only the log function."""
        mock_log = MagicMock()
        original_experiment = litlogger.experiment

        set_global(log=mock_log)

        assert litlogger.log is mock_log
        assert litlogger.experiment is original_experiment

    def test_set_log_metrics(self):
        """Test setting only the log_metrics function."""
        mock_log_metrics = MagicMock()

        set_global(log_metrics=mock_log_metrics)

        assert litlogger.log_metrics is mock_log_metrics

    def test_set_log_file(self):
        """Test setting only the log_file function."""
        mock_log_file = MagicMock()

        set_global(log_file=mock_log_file)

        assert litlogger.log_file is mock_log_file

    def test_set_finalize(self):
        """Test setting only the finalize function."""
        mock_finalize = MagicMock()

        set_global(finalize=mock_finalize)

        assert litlogger.finalize is mock_finalize

    def test_set_all_globals(self):
        """Test setting all globals at once."""
        mock_experiment = MagicMock()
        mock_log = MagicMock()
        mock_log_metrics = MagicMock()
        mock_log_file = MagicMock()
        mock_finalize = MagicMock()

        set_global(
            experiment=mock_experiment,
            log=mock_log,
            log_metrics=mock_log_metrics,
            log_file=mock_log_file,
            finalize=mock_finalize,
        )

        assert litlogger.experiment is mock_experiment
        assert litlogger.log is mock_log
        assert litlogger.log_metrics is mock_log_metrics
        assert litlogger.log_file is mock_log_file
        assert litlogger.finalize is mock_finalize

    def test_set_none_values_does_nothing(self):
        """Test that passing None values doesn't change globals."""
        original_experiment = litlogger.experiment
        original_log = litlogger.log

        set_global(experiment=None, log=None)

        # Globals should remain unchanged
        assert litlogger.experiment is original_experiment
        assert litlogger.log is original_log


class TestUnsetGlobals:
    """Test the unset_globals function."""

    def test_unset_resets_experiment_to_none(self):
        """Test that unset_globals sets experiment to None."""
        # Set a mock experiment first
        litlogger.experiment = MagicMock()

        unset_globals()

        assert litlogger.experiment is None

    def test_unset_creates_preinit_callables(self):
        """Test that unset_globals creates PreInitCallable wrappers."""
        unset_globals()

        # All functions should be PreInitCallable instances (callables)
        assert callable(litlogger.log)
        assert callable(litlogger.log_metrics)
        assert callable(litlogger.log_file)
        assert callable(litlogger.finalize)

    def test_unset_log_raises_on_call(self):
        """Test that calling log after unset raises RuntimeError."""
        unset_globals()

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before litlogger.log\\(\\)"):
            litlogger.log({"loss": 0.5})

    def test_unset_log_metrics_raises_on_call(self):
        """Test that calling log_metrics after unset raises RuntimeError."""
        unset_globals()

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before litlogger.log_metrics\\(\\)"):
            litlogger.log_metrics({"loss": 0.5})

    def test_unset_log_file_raises_on_call(self):
        """Test that calling log_file after unset raises RuntimeError."""
        unset_globals()

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before litlogger.log_file\\(\\)"):
            litlogger.log_file("test.txt")

    def test_unset_finalize_raises_on_call(self):
        """Test that calling finalize after unset raises RuntimeError."""
        unset_globals()

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before litlogger.finalize\\(\\)"):
            litlogger.finalize()

    def test_unset_after_set_restores_preinit_state(self):
        """Test that unset_globals can restore state after setting globals."""
        # Set all globals
        mock_experiment = MagicMock()
        mock_log = MagicMock()

        set_global(experiment=mock_experiment, log=mock_log)

        # Verify they're set
        assert litlogger.experiment is mock_experiment
        assert litlogger.log is mock_log

        # Unset
        unset_globals()

        # Verify they're reset to preinit state
        assert litlogger.experiment is None

        with pytest.raises(RuntimeError):
            litlogger.log({"loss": 0.5})
