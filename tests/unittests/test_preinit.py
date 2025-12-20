# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for litlogger._preinit pre-initialization wrappers."""

import pytest
from litlogger._preinit import PreInitObject, pre_init_callable


class TestPreInitObject:
    """Test the PreInitObject class."""

    def test_init_with_name_only(self):
        """Test initialization with just a name."""
        obj = PreInitObject("test.object")
        assert obj._name == "test.object"

    def test_init_with_destination_copies_doc(self):
        """Test that __doc__ is copied from destination."""

        def sample_func():
            """Sample documentation."""

        obj = PreInitObject("test.object", destination=sample_func)
        assert obj.__doc__ == "Sample documentation."

    def test_getitem_raises_error(self):
        """Test that accessing items raises RuntimeError."""
        obj = PreInitObject("test.object")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.object\\['key'\\]"):
            _ = obj["key"]

    def test_setitem_raises_error(self):
        """Test that setting items raises RuntimeError."""
        obj = PreInitObject("test.object")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.object\\['key'\\]"):
            obj["key"] = "value"

    def test_setattr_public_raises_error(self):
        """Test that setting public attributes raises RuntimeError."""
        obj = PreInitObject("test.object")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.object.attr"):
            obj.attr = "value"

    def test_setattr_private_works(self):
        """Test that setting private attributes (starting with _) works."""
        obj = PreInitObject("test.object")
        obj._private_attr = "value"
        assert obj._private_attr == "value"

    def test_getattr_public_raises_error(self):
        """Test that accessing public attributes raises RuntimeError."""
        obj = PreInitObject("test.object")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.object.attr"):
            _ = obj.attr

    def test_getattr_private_raises_attribute_error(self):
        """Test that accessing non-existent private attributes raises AttributeError."""
        obj = PreInitObject("test.object")

        with pytest.raises(AttributeError):
            _ = obj._nonexistent

    def test_different_names_in_error_messages(self):
        """Test that error messages include the correct object name."""
        obj1 = PreInitObject("litlogger.experiment")
        obj2 = PreInitObject("litlogger.logger")

        with pytest.raises(RuntimeError, match="litlogger.experiment.attr"):
            _ = obj1.attr

        with pytest.raises(RuntimeError, match="litlogger.logger.attr"):
            _ = obj2.attr


class TestPreInitCallable:
    """Test the pre_init_callable function."""

    def test_creates_callable(self):
        """Test that pre_init_callable returns a callable."""
        func = pre_init_callable("test.func")
        assert callable(func)

    def test_callable_raises_on_call(self):
        """Test that calling the returned function raises RuntimeError."""
        func = pre_init_callable("test.func")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.func\\(\\)"):
            func()

    def test_callable_raises_with_args(self):
        """Test that calling with args still raises RuntimeError."""
        func = pre_init_callable("test.func")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.func\\(\\)"):
            func("arg1", "arg2")

    def test_callable_raises_with_kwargs(self):
        """Test that calling with kwargs still raises RuntimeError."""
        func = pre_init_callable("test.func")

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before test.func\\(\\)"):
            func(key="value")

    def test_name_is_set(self):
        """Test that the function name is set correctly."""
        func = pre_init_callable("litlogger.log_metrics")
        assert func.__name__ == "litlogger.log_metrics"

    def test_doc_copied_from_destination(self):
        """Test that __doc__ is copied from destination."""

        def sample_func():
            """This is sample documentation."""

        func = pre_init_callable("test.func", destination=sample_func)
        assert func.__doc__ == "This is sample documentation."

    def test_wrapped_attribute_set(self):
        """Test that __wrapped__ is set when destination is provided."""

        def sample_func():
            pass

        func = pre_init_callable("test.func", destination=sample_func)
        assert func.__wrapped__ is sample_func

    def test_without_destination(self):
        """Test creating pre_init_callable without destination."""
        func = pre_init_callable("test.func")
        assert func.__name__ == "test.func"
        # Should not have __doc__ or __wrapped__ set (or they're None)
        assert not hasattr(func, "__wrapped__") or func.__wrapped__ is None

    def test_different_names_in_error_messages(self):
        """Test that error messages include the correct function name."""
        func1 = pre_init_callable("litlogger.log")
        func2 = pre_init_callable("litlogger.finalize")

        with pytest.raises(RuntimeError, match="litlogger.log\\(\\)"):
            func1()

        with pytest.raises(RuntimeError, match="litlogger.finalize\\(\\)"):
            func2()
