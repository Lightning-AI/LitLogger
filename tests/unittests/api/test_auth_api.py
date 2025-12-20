# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for auth API."""

from unittest.mock import MagicMock, patch

from litlogger.api.auth_api import AuthApi


class TestAuthApi:
    """Test the AuthApi class."""

    def test_init(self):
        """Test initialization creates Auth instance."""
        with patch("litlogger.api.auth_api.Auth") as mock_auth_class:
            api = AuthApi()
            mock_auth_class.assert_called_once()
            assert api.auth == mock_auth_class.return_value

    def test_authenticate(self):
        """Test authenticate calls auth.authenticate()."""
        with patch("litlogger.api.auth_api.Auth") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth_class.return_value = mock_auth

            api = AuthApi()
            api.authenticate()

            mock_auth.authenticate.assert_called_once()
