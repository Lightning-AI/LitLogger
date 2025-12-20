# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for color utilities."""

from litlogger.colors import _create_colors


class TestCreateColors:
    """Test the _create_colors function."""

    def test_same_name_different_version_different_colors(self):
        """Test that same name with different versions get different colors."""
        color1 = _create_colors("my-experiment", "v1")
        color2 = _create_colors("my-experiment", "v2")
        color3 = _create_colors("my-experiment", "v3")

        # All should be different
        assert color1 != color2
        assert color2 != color3
        assert color1 != color3

    def test_deterministic_colors(self):
        """Test that same name+version always returns same colors."""
        color1 = _create_colors("my-experiment", "v1")
        color2 = _create_colors("my-experiment", "v1")

        assert color1 == color2

    def test_returns_tuple_of_two_colors(self):
        """Test that function returns a tuple of (light_color, dark_color)."""
        result = _create_colors("test", "v1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].startswith("#")
        assert result[1].startswith("#")

    def test_different_names_different_colors(self):
        """Test that different experiment names get different colors."""
        color1 = _create_colors("experiment-a", "v1")
        color2 = _create_colors("experiment-b", "v1")

        # Should be different (with high probability due to hash)
        assert color1 != color2

    def test_fallback_to_random_without_name_version(self):
        """Test that without name/version, colors are still returned."""
        color1 = _create_colors()
        color2 = _create_colors(None, None)

        # Should return valid colors (may or may not be same due to random)
        assert isinstance(color1, tuple)
        assert len(color1) == 2
        assert isinstance(color2, tuple)
        assert len(color2) == 2
