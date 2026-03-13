# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for File, Image, and Text media types."""

import os
import tempfile

import pytest
from litlogger.media import File, Image, Text
from litlogger.types import MediaType


class TestFileInit:
    """Test File construction and basic attributes."""

    def test_init_with_path(self):
        f = File("path/to/file.txt")
        assert f.path == "path/to/file.txt"
        assert f.name == ""
        assert f.description == ""
        assert f._temp_path is None
        assert f._download_fn is None

    def test_init_with_description(self):
        f = File("data.csv", description="training data")
        assert f.description == "training data"

    def test_name_defaults_empty(self):
        """Name is empty until the experiment assigns it."""
        f = File("data.csv")
        assert f.name == ""

    def test_name_is_assignable(self):
        f = File("data.csv")
        f.name = "my-key"
        assert f.name == "my-key"

    def test_media_type(self):
        assert File("x")._media_type == MediaType.FILE


class TestFileUploadPath:
    """Test _get_upload_path hardlink / copy / fallback behavior."""

    def test_nonexistent_file_returns_original_path(self):
        f = File("does/not/exist.txt")
        assert f._get_upload_path() == "does/not/exist.txt"
        assert f._temp_path is None

    def test_empty_path_returns_empty(self):
        f = File("")
        assert f._get_upload_path() == ""

    def test_existing_file_creates_temp_copy(self):
        """Upload path should be a temp file (hardlink or copy) distinct from the original."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"hello")
            original = tmp.name
        try:
            f = File(original)
            upload = f._get_upload_path()

            assert upload != original
            assert os.path.exists(upload)
            assert f._temp_path == upload
            with open(upload) as fh:
                assert fh.read() == "hello"
        finally:
            f._cleanup()
            os.unlink(original)

    def test_preserves_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"a,b")
            original = tmp.name
        try:
            f = File(original)
            upload = f._get_upload_path()
            assert upload.endswith(".csv")
        finally:
            f._cleanup()
            os.unlink(original)

    def test_original_can_be_deleted_after_upload_path(self):
        """The upload path should survive deletion of the original file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"data")
            original = tmp.name
        try:
            f = File(original)
            upload = f._get_upload_path()
            os.unlink(original)

            assert os.path.exists(upload)
            with open(upload) as fh:
                assert fh.read() == "data"
        finally:
            f._cleanup()


class TestFileCleanup:
    """Test _cleanup removes temp files."""

    def test_cleanup_no_temp(self):
        f = File("test.txt")
        f._cleanup()  # Should not raise

    def test_cleanup_removes_temp(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"x")
            original = tmp.name
        try:
            f = File(original)
            upload = f._get_upload_path()
            assert os.path.exists(upload)

            f._cleanup()
            assert not os.path.exists(upload)
            assert f._temp_path is None
        finally:
            if os.path.exists(original):
                os.unlink(original)

    def test_cleanup_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"x")
            original = tmp.name
        try:
            f = File(original)
            f._get_upload_path()
            f._cleanup()
            f._cleanup()  # Second call should not raise
        finally:
            if os.path.exists(original):
                os.unlink(original)


class TestFileSave:
    """Test File.save download behavior."""

    def test_save_without_remote_context_raises(self):
        f = File("test.txt")
        with pytest.raises(RuntimeError, match="no remote context"):
            f.save("/tmp/out.txt")

    def test_save_delegates_to_download_fn(self):
        f = File("test.txt")
        f._download_fn = lambda path: path + ".downloaded"

        result = f.save("/tmp/out.txt")
        assert result == "/tmp/out.txt.downloaded"


class TestFileProtocol:
    """Test repr, equality, hashing."""

    def test_repr(self):
        assert repr(File("test.txt")) == "File('test.txt')"

    def test_equality_same_path(self):
        assert File("a.txt") == File("a.txt")

    def test_inequality_different_path(self):
        assert File("a.txt") != File("b.txt")

    def test_inequality_with_subclass(self):
        assert File("a.txt") != Image("a.txt")

    def test_inequality_with_non_file(self):
        assert File("a.txt").__eq__("a.txt") is NotImplemented

    def test_hash_consistent(self):
        f1 = File("a.txt")
        f2 = File("a.txt")
        assert hash(f1) == hash(f2)
        assert {f1, f2} == {f1}

    def test_hash_differs_across_types(self):
        assert hash(File("a.txt")) != hash(Image("a.txt"))


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class TestImageInit:
    """Test Image construction."""

    def test_from_path(self):
        i = Image("photo.png")
        assert i.path == "photo.png"
        assert i._data == "photo.png"
        assert i._format == "png"

    def test_from_path_with_description(self):
        i = Image("photo.png", description="a photo")
        assert i.description == "a photo"

    def test_from_object_sets_empty_path(self):
        """Non-string data should set path='' until rendered."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")
        pil = PILImage.new("RGB", (8, 8))
        i = Image(pil)
        assert i.path == ""

    def test_custom_format(self):
        i = Image("photo.jpg", format="jpeg")
        assert i._format == "jpeg"

    def test_media_type(self):
        assert Image("x.png")._media_type == MediaType.IMAGE


class TestImageUploadPath:
    """Test Image._get_upload_path for different data types."""

    def test_string_path_nonexistent_returns_path(self):
        i = Image("nonexistent.png")
        assert i._get_upload_path() == "nonexistent.png"

    def test_pil_image_renders_to_temp(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")

        pil = PILImage.new("RGB", (16, 16), color=(255, 0, 0))
        i = Image(pil)
        path = i._get_upload_path()

        assert path.endswith(".png")
        assert os.path.exists(path)
        reopened = PILImage.open(path)
        assert reopened.size == (16, 16)
        i._cleanup()

    def test_numpy_hwc_uint8(self):
        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("numpy/PIL not available")

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        i = Image(arr)
        path = i._get_upload_path()

        assert os.path.exists(path)
        reopened = PILImage.open(path)
        assert reopened.size == (32, 32)
        i._cleanup()

    def test_numpy_float_normalized(self):
        """Float arrays with max <= 1.0 should be scaled to 0-255."""
        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("numpy/PIL not available")

        arr = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        i = Image(arr)
        path = i._get_upload_path()
        reopened = PILImage.open(path)
        assert reopened.size == (8, 8)
        i._cleanup()

    def test_numpy_grayscale(self):
        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("numpy/PIL not available")

        arr = np.zeros((16, 16), dtype=np.uint8)
        i = Image(arr)
        path = i._get_upload_path()
        reopened = PILImage.open(path)
        assert reopened.size == (16, 16)
        i._cleanup()

    def test_numpy_chw_transposed(self):
        """CHW format (e.g. 3,H,W) should be transposed to HWC."""
        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("numpy/PIL not available")

        arr = np.zeros((3, 24, 32), dtype=np.uint8)
        i = Image(arr)
        path = i._get_upload_path()
        reopened = PILImage.open(path)
        assert reopened.size == (32, 24)  # width, height
        i._cleanup()

    def test_numpy_single_channel_squeezed(self):
        """Single-channel 3D array (H,W,1) should be squeezed."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        arr = np.zeros((16, 16, 1), dtype=np.uint8)
        i = Image(arr)
        path = i._get_upload_path()
        assert os.path.exists(path)
        i._cleanup()

    def test_numpy_unsupported_ndim(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        arr = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        i = Image(arr)
        with pytest.raises(ValueError, match="Unsupported array shape"):
            i._get_upload_path()

    def test_unsupported_type_raises(self):
        i = Image({"not": "an image"})
        with pytest.raises(TypeError, match="Unsupported image type"):
            i._get_upload_path()

    def test_custom_format_suffix(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")

        pil = PILImage.new("RGB", (8, 8))
        i = Image(pil, format="bmp")
        path = i._get_upload_path()
        assert path.endswith(".bmp")
        i._cleanup()


class TestImageCleanup:
    """Test Image._cleanup removes rendered temp files."""

    def test_cleanup_removes_rendered_temp(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")

        pil = PILImage.new("RGB", (8, 8))
        i = Image(pil)
        path = i._get_upload_path()
        assert os.path.exists(path)

        i._cleanup()
        assert not os.path.exists(path)
        assert i._temp_path is None

    def test_cleanup_no_render_is_noop(self):
        """Cleanup before rendering should not raise."""
        i = Image("test.png")
        i._cleanup()

    def test_cleanup_idempotent(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")

        pil = PILImage.new("RGB", (8, 8))
        i = Image(pil)
        i._get_upload_path()
        i._cleanup()
        i._cleanup()  # second call should not raise

    def test_cleanup_numpy_rendered(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        i = Image(arr)
        path = i._get_upload_path()
        assert os.path.exists(path)

        i._cleanup()
        assert not os.path.exists(path)


class TestImageRepr:
    def test_path_repr(self):
        assert repr(Image("test.png")) == "Image('test.png')"

    def test_object_repr_before_render(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")
        pil = PILImage.new("RGB", (8, 8))
        assert repr(Image(pil)) == "Image('')"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TestTextInit:
    """Test Text construction."""

    def test_init(self):
        t = Text("hello world")
        assert t._content == "hello world"
        assert t.path == ""

    def test_init_with_description(self):
        t = Text("hello", description="greeting")
        assert t.description == "greeting"

    def test_media_type(self):
        assert Text("x")._media_type == MediaType.TEXT


class TestTextUploadPath:
    """Test Text._get_upload_path rendering."""

    def test_renders_to_temp_file(self):
        t = Text("hello world")
        path = t._get_upload_path()

        assert path.endswith(".txt")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "hello world"
        t._cleanup()

    def test_sets_path_after_render(self):
        t = Text("content")
        assert t.path == ""
        path = t._get_upload_path()
        assert t.path == path
        t._cleanup()

    def test_reuses_rendered_path(self):
        t = Text("content")
        p1 = t._get_upload_path()
        p2 = t._get_upload_path()
        # After first render, path is set and file exists, so super()._get_upload_path()
        # creates a hardlink. Both should have the same content.
        with open(p1, encoding="utf-8") as f1, open(p2, encoding="utf-8") as f2:
            assert f1.read() == f2.read()
        t._cleanup()
        # Clean up the hardlinked temp from second call
        if p1 != p2 and os.path.exists(p1):
            os.unlink(p1)

    def test_unicode_content(self):
        t = Text("unicode: \u2603 \u2764 \U0001f680")
        path = t._get_upload_path()
        with open(path, encoding="utf-8") as f:
            assert f.read() == "unicode: \u2603 \u2764 \U0001f680"
        t._cleanup()

    def test_empty_string(self):
        t = Text("")
        path = t._get_upload_path()
        with open(path, encoding="utf-8") as f:
            assert f.read() == ""
        t._cleanup()

    def test_multiline_content(self):
        content = "line 1\nline 2\nline 3"
        t = Text(content)
        path = t._get_upload_path()
        with open(path, encoding="utf-8") as f:
            assert f.read() == content
        t._cleanup()


class TestTextCleanup:
    """Test Text._cleanup removes rendered temp files."""

    def test_cleanup_removes_rendered_temp(self):
        t = Text("hello")
        path = t._get_upload_path()
        assert os.path.exists(path)

        t._cleanup()
        assert not os.path.exists(path)
        assert t._temp_path is None

    def test_cleanup_before_render_is_noop(self):
        t = Text("hello")
        t._cleanup()  # should not raise

    def test_cleanup_idempotent(self):
        t = Text("hello")
        t._get_upload_path()
        t._cleanup()
        t._cleanup()  # second call should not raise


class TestTextRepr:
    def test_repr_before_render(self):
        assert repr(Text("hello")) == "Text('')"

    def test_repr_after_render(self):
        t = Text("hello")
        t._get_upload_path()
        assert "Text(" in repr(t)
        assert t.path in repr(t)
        t._cleanup()
