"""Tiny compatibility fix for Streamlit camera captures.

Streamlit's camera_input returns an UploadedFile-like object. CropGuard wraps
its captured bytes in _CameraCapture, which exposes getvalue() but is not a
seekable file object. Pillow expects seek/read for Image.open(). This shim
converts only getvalue()-based objects to BytesIO before Pillow opens them.
No model, inference, routes, or training behavior is changed.
"""

from io import BytesIO

from PIL import Image as _Image

_original_open = _Image.open


def _camera_safe_open(fp, *args, **kwargs):
    if hasattr(fp, "getvalue") and not hasattr(fp, "read"):
        return _original_open(BytesIO(fp.getvalue()), *args, **kwargs)
    return _original_open(fp, *args, **kwargs)


_Image.open = _camera_safe_open
