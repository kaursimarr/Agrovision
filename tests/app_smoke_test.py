"""Smoke test for app.py WITHOUT Streamlit or TensorFlow installed.

It runs app.py top-to-bottom against a stand-in `streamlit` module and a fake Predictor, in several
scenarios (single photo, several photos, no photo, no model). It catches Python errors (wrong names,
bad arguments, logic bugs); it cannot check how the real Streamlit renders the page.

    python tests/app_smoke_test.py
"""
from __future__ import annotations

import io
import json
import runpy
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

NAMES = json.loads((ROOT / "tests" / "class_names_38.json").read_text())


class StopApp(Exception):
    pass


class FakeFile:
    def __init__(self, data: bytes, name: str):
        self._d, self.name = data, name

    def getvalue(self):
        return self._d


class Ctx:
    """Stand-in for containers / columns / tabs / placeholders."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):          # any widget method: delegate to the module-level function
        return getattr(ST, name)


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def _png(seed: int, size=(300, 260)) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


class FakeST(types.ModuleType):
    def __init__(self, scenario: dict):
        super().__init__("streamlit")
        self.scenario = scenario
        self.session_state = SessionState()
        self.sidebar = Ctx()
        self.calls: dict[str, int] = {}

    def __getattr__(self, name):
        raise AttributeError(f"app.py used st.{name}, which the smoke-test stand-in does not define "
                             f"(add it if it is a real Streamlit API)")


def make_st(scenario: dict) -> FakeST:
    st = FakeST(scenario)
    globals()["ST"] = st          # Ctx delegates widget calls to the current stand-in module

    def count(name):
        def wrapper(*a, **k):
            st.calls[name] = st.calls.get(name, 0) + 1
            return None
        return wrapper

    for n in ("set_page_config", "markdown", "write", "caption", "info", "warning", "error", "success", "divider",
              "header", "subheader", "image", "dataframe", "bar_chart", "metric", "progress", "download_button",
              "rerun"):
        setattr(st, n, count(n))
    st.button = lambda *a, **k: False
    st.text_input = lambda *a, **k: scenario.get("query", "")
    st.multiselect = lambda *a, **k: scenario.get("multiselect", [])
    st.camera_input = lambda *a, **k: None
    st.toggle = lambda *a, **k: scenario.get("toggle", True)
    st.checkbox = lambda label, key=None, **k: bool(st.session_state.get(key, False))
    st.expander = lambda *a, **k: Ctx()
    st.spinner = lambda *a, **k: Ctx()
    st.empty = lambda: Ctx()
    st.container = lambda *a, **k: Ctx()
    st.columns = lambda spec, **k: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [Ctx() for _ in labels]

    def stop():
        raise StopApp()
    st.stop = stop

    def selectbox(label, options, index=0, format_func=str, key=None, **k):
        opts = list(options)
        [format_func(o) for o in opts]
        return opts[index] if opts else None
    st.selectbox = selectbox

    def radio(label, options, index=0, format_func=str, horizontal=False, key=None, **k):
        opts = list(options)
        [format_func(o) for o in opts]
        return opts[0]
    st.radio = radio

    def slider(label, min_value=0, max_value=100, value=None, *a, **k):
        return value if value is not None else min_value
    st.slider = slider

    st.file_uploader = lambda *a, **k: scenario.get("files", [])

    def passthrough(*dargs, **dkwargs):
        if dargs and callable(dargs[0]):
            return dargs[0]
        return lambda fn: fn
    st.cache_resource = passthrough
    st.cache_data = passthrough
    return st


def build_project(tmp: Path, with_models: bool) -> None:
    for item in ("app.py", "src"):
        src = ROOT / item
        (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp / item, **({"ignore": shutil.ignore_patterns("__pycache__")} if src.is_dir() else {}))
    # fake trained artefacts
    (tmp / "outputs" / "comparison").mkdir(parents=True)
    (tmp / "outputs" / "class_names.json").write_text(json.dumps(NAMES))
    (tmp / "models").mkdir()
    if with_models:
        for m in ("cnn", "mobilenetv2", "resnet50", "vit"):
            (tmp / "models" / f"{m}.keras").write_bytes(b"x")
            d = tmp / "outputs" / m
            d.mkdir()
            (d / "metrics.json").write_text("{}")
            (d / "training_curves.png").write_bytes(b"x")
            (d / "confusion_matrix.png").write_bytes(b"x")
            (d / "confusion_matrix_normalized.png").write_bytes(b"x")
            rng = np.random.default_rng(1)
            rep = pd.DataFrame({"precision": rng.random(41), "recall": rng.random(41),
                                "f1-score": rng.random(41), "support": 100},
                               index=NAMES + ["accuracy", "macro avg", "weighted avg"])
            rep.to_csv(d / "classification_report.csv")
    pd.DataFrame([
        ["Custom CNN (from scratch)", .8991, .8977, .9072, .8974, .8986, .9885, 1.25, 15.2, 7.4, 17, 67.3],
        ["MobileNetV2 (transfer learning)", .9426, .9461, .9442, .9439, .9422, .9912, 2.31, 22.3, 2.2, 13, 11.4],
        ["ResNet50 (transfer learning)", .9643, .965, .9638, .9634, .9645, .9975, 23.67, 211.4, 5.2, 13, 29.8],
        ["Vision Transformer (ViT from scratch)", .9125, .9140, .9112, .9120, .9135, .9902, 0.68, 2.6, 3.8, 20, 18.5],
    ], columns=["Model", "Test accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)",
                "Top-3 accuracy", "Params (M)", "Size (MB)", "Inference (ms/img)", "Epochs run", "Train time (min)"]
    ).to_csv(tmp / "outputs" / "comparison" / "metrics_comparison.csv", index=False)
    (tmp / "outputs" / "comparison" / "metrics_comparison.png").write_bytes(b"x")


def install_fake_predictor() -> None:
    mod = types.ModuleType("src.predict")
    from src.labels import parse_label

    class Predictor:
        def __init__(self, model_file):
            self.model_file = model_file

        def predict(self, image, top_k=3):
            assert isinstance(image, Image.Image)
            seed = sum(image.tobytes()[:2000]) + len(str(self.model_file))
            rng = np.random.default_rng(seed)
            p = rng.dirichlet(np.ones(len(NAMES)) * 0.1)
            out = []
            for i in np.argsort(p)[::-1][:top_k]:
                plant, cond, healthy = parse_label(NAMES[i])
                out.append({"class_name": NAMES[i], "plant": plant, "condition": cond, "healthy": healthy,
                            "confidence": float(p[i])})
            return out

    mod.Predictor = Predictor
    sys.modules["src.predict"] = mod


def run_scenario(title: str, scenario: dict, with_models: bool = True) -> dict:
    tmp = Path(tempfile.mkdtemp())
    build_project(tmp, with_models)
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    sys.path.insert(0, str(tmp))
    st = make_st(scenario)
    sys.modules["streamlit"] = st
    try:
        import src  # noqa: F401  (from the temp copy)
        install_fake_predictor()
        try:
            runpy.run_path(str(tmp / "app.py"), run_name="__main__")
        except StopApp:
            pass
    finally:
        sys.path.remove(str(tmp))
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ok  - {title}: {sum(st.calls.values())} UI calls, history={len(st.session_state['history'])}")
    return {"st": st}


def main() -> None:
    one = [FakeFile(_png(1), "leaf1.png")]
    many = [FakeFile(_png(1), "leaf1.png"), FakeFile(_png(2), "leaf2.jpg"), FakeFile(b"not an image", "bad.txt")]
    r = run_scenario("single photo + compare models", {"files": one, "toggle": True})
    assert r["st"].calls["download_button"] >= 2 and len(r["st"].session_state["history"]) == 1
    r = run_scenario("three files, one unreadable", {"files": many, "toggle": False})
    assert len(r["st"].session_state["history"]) == 2          # bad file skipped, 2 good ones recorded
    run_scenario("no photo uploaded", {"files": []})
    run_scenario("library search + filters", {"files": [], "query": "blight", "multiselect": ["Tomato"]})
    run_scenario("no trained models", {"files": one}, with_models=False)
    print("All app smoke tests passed.")


if __name__ == "__main__":
    main()
