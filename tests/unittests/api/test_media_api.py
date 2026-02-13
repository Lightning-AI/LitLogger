# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for media API."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi import V1MediaType
from litlogger.api.media_api import MediaApi


class TestMediaApi:
    """Test the MediaApi class."""

    def test_init_with_client(self):
        """Test initialization with a custom client."""
        mock_client = MagicMock()
        api = MediaApi(client=mock_client)
        assert api.client is mock_client

    def test_init_without_client(self):
        """Test initialization creates a default client."""
        with patch("litlogger.api.media_api.LitRestClient") as mock_client_class:
            api = MediaApi()
            mock_client_class.assert_called_once_with(max_retries=5)
            assert api.client == mock_client_class.return_value

    def test_upload_media_success(self):
        """Test successful media upload with create, upload, and update calls."""
        mock_client = MagicMock()
        api = MediaApi(client=mock_client)

        mock_teamspace = MagicMock()
        mock_teamspace.id = "ts-123"
        mock_teamspace._teamspace_api = MagicMock()

        media = SimpleNamespace(storage_path="media/abc.png", id="media-1", cluster_id="acc-456")
        create_response = SimpleNamespace(media=media, already_exists=False)
        mock_client.lit_logger_service_create_lit_logger_media.return_value = create_response

        updated_media = SimpleNamespace(id="media-1", storage_path="media/abc.png")
        mock_client.lit_logger_service_update_lit_logger_media.return_value = updated_media

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
            f.write(b"fake-image-bytes")
            temp_file = f.name

        try:
            with patch("litlogger.api.media_api._compute_file_hash", return_value="hash"):
                result = api.upload_media(
                    experiment_id="exp-789",
                    teamspace=mock_teamspace,
                    file_path=temp_file,
                    name="image",
                    media_type=V1MediaType.IMAGE,
                    step=5,
                    epoch=2,
                    caption="caption",
                )

            assert result is updated_media
            mock_client.lit_logger_service_create_lit_logger_media.assert_called_once()
            mock_teamspace._teamspace_api.upload_file.assert_called_once()
            mock_client.lit_logger_service_update_lit_logger_media.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_upload_media_already_exists_skips_upload(self):
        """Test that upload and update are skipped when media already exists."""
        mock_client = MagicMock()
        api = MediaApi(client=mock_client)

        mock_teamspace = MagicMock()
        mock_teamspace.id = "ts-123"
        mock_teamspace._teamspace_api = MagicMock()

        media = SimpleNamespace(storage_path="media/abc.png", id="media-1", cluster_id="acc-456")
        create_response = SimpleNamespace(media=media, already_exists=True)
        mock_client.lit_logger_service_create_lit_logger_media.return_value = create_response

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
            f.write(b"fake-image-bytes")
            temp_file = f.name

        try:
            with patch("litlogger.api.media_api._compute_file_hash", return_value="hash"):
                result = api.upload_media(
                    experiment_id="exp-789",
                    teamspace=mock_teamspace,
                    file_path=temp_file,
                    name="image",
                    media_type=V1MediaType.IMAGE,
                )

            assert result is media
            mock_teamspace._teamspace_api.upload_file.assert_not_called()
            mock_client.lit_logger_service_update_lit_logger_media.assert_not_called()
        finally:
            os.unlink(temp_file)

    def test_upload_media_file_not_found(self):
        """Test upload with non-existent file."""
        api = MediaApi(client=MagicMock())
        mock_teamspace = MagicMock()
        mock_teamspace.id = "ts-123"
        mock_teamspace._teamspace_api = MagicMock()

        with pytest.raises(FileNotFoundError, match="file not found"):
            api.upload_media(
                experiment_id="exp-789",
                teamspace=mock_teamspace,
                file_path="/nonexistent/file.png",
                name="image",
                media_type=V1MediaType.IMAGE,
            )
