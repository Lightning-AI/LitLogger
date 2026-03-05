# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for color utilities."""

from litlogger.colors import _create_colors


class TestCreateColors:
    """Test the _create_colors function."""

    def test_different_names_different_colors(self):
        """Test that different experiment names get different colors."""
        color1 = _create_colors("experiment-a")
        color2 = _create_colors("experiment-b")

        # Should be different (with high probability due to hash)
        assert color1 != color2

    def test_deterministic_colors(self):
        """Test that same name always returns same colors."""
        color1 = _create_colors("my-experiment")
        color2 = _create_colors("my-experiment")

        assert color1 == color2

    def test_returns_tuple_of_two_colors(self):
        """Test that function returns a tuple of (light_color, dark_color)."""
        result = _create_colors("test")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].startswith("#")
        assert result[1].startswith("#")

    def test_fallback_to_random_without_name(self):
        """Test that without name, colors are still returned."""
        color1 = _create_colors()
        color2 = _create_colors(None)

        # Should return valid colors (may or may not be same due to random)
        assert isinstance(color1, tuple)
        assert len(color1) == 2
        assert isinstance(color2, tuple)
        assert len(color2) == 2
