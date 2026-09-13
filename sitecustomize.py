"""Small runtime compatibility hooks for the CropGuard Streamlit frontend.

The Pillow hook preserves camera uploads. The Streamlit hook only provides a
fallback treatment advisory after the prediction table if the normal advisory
container is skipped by a rendering failure. It never changes model,
inference, training, routes, or prediction values.
"""

from io import BytesIO
import builtins
import json
import re
import sys
from pathlib import Path

from PIL import Image as _Image

# Existing camera compatibility: make getvalue()-only camera captures readable
# by Pillow without changing normal uploaded-file behavior.
_original_open = _Image.open


def _camera_safe_open(fp, *args, **kwargs):
    if hasattr(fp, "getvalue") and not hasattr(fp, "read"):
        return _original_open(BytesIO(fp.getvalue()), *args, **kwargs)
    return _original_open(fp, *args, **kwargs)


_Image.open = _camera_safe_open

_original_import = builtins.__import__
_streamlit_patched = False


def _advisory_for(label: str) -> dict:
    path = Path(__file__).resolve().parent / "backend" / "treatments.json"
    try:
        treatments = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        treatments = {}

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    wanted = {norm(label), norm(label.replace("___", "_"))}
    disease = label.split("___", 1)[-1] if "___" in label else label
    wanted.add(norm(disease))
    for key, advisory in treatments.items():
        if norm(key) in wanted:
            return advisory if isinstance(advisory, dict) else {}
    return {}


def _patch_streamlit(module) -> None:
    global _streamlit_patched
    if _streamlit_patched or module is None or not hasattr(module, "table"):
        return
    original_table = module.table

    def table(data, *args, **kwargs):
        result = original_table(data, *args, **kwargs)
        try:
            # CropGuard's prediction table has exactly these display keys.
            if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
                if all("class" in row and "probability" in row for row in data):
                    advisory = _advisory_for(str(data[0]["class"]))
                    module.markdown("### 🩺 Disease fix & treatment")
                    module.markdown(
                        "**What to do:** "
                        + str(advisory.get("summary") or "Follow the recommended disease-management steps below.")
                    )
                    actions = advisory.get("actions") or []
                    if isinstance(actions, list) and actions:
                        for action in actions:
                            module.markdown(f"- {action}")
                    else:
                        module.info("No class-specific treatment steps are available for this result.")
                    for source in advisory.get("sources") or []:
                        module.caption(str(source))
        except Exception:
            # Advisory display must never affect the actual prediction result.
            pass
        return result

    module.table = table
    _streamlit_patched = True


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "streamlit" or name.startswith("streamlit."):
        try:
            _patch_streamlit(sys.modules.get("streamlit"))
        except Exception:
            pass
    return module


builtins.__import__ = _import
