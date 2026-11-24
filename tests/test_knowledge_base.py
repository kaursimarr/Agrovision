"""Checks that every PlantVillage class maps to the right knowledge-base entry (no TensorFlow).
Run from the project root:  python tests/test_knowledge_base.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.disease_info import _DISEASES, SEVERITY_ORDER, get_info  # noqa: E402

PLANTVILLAGE_CLASSES = """Apple___Apple_scab Apple___Black_rot Apple___Cedar_apple_rust Apple___healthy
Blueberry___healthy Cherry_(including_sour)___Powdery_mildew Cherry_(including_sour)___healthy
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot Corn_(maize)___Common_rust_ Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy Grape___Black_rot Grape___Esca_(Black_Measles) Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy Orange___Haunglongbing_(Citrus_greening) Peach___Bacterial_spot Peach___healthy
Pepper,_bell___Bacterial_spot Pepper,_bell___healthy Potato___Early_blight Potato___Late_blight Potato___healthy
Raspberry___healthy Soybean___healthy Squash___Powdery_mildew Strawberry___Leaf_scorch Strawberry___healthy
Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Leaf_Mold Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite Tomato___Target_Spot Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus Tomato___healthy"""

# the two Corn / Tomato names contain a space inside the folder name; rebuild the list properly
NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

EXPECTED = {  # class -> expected knowledge-base name fragment
    "Apple___Apple_scab": "Apple scab", "Apple___Black_rot": "Apple black rot",
    "Apple___Cedar_apple_rust": "Cedar apple rust", "Cherry_(including_sour)___Powdery_mildew": "Cherry powdery",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "gray leaf spot",
    "Corn_(maize)___Common_rust_": "common rust", "Corn_(maize)___Northern_Leaf_Blight": "Northern corn leaf blight",
    "Grape___Black_rot": "Grape black rot", "Grape___Esca_(Black_Measles)": "esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape leaf blight",
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus greening", "Peach___Bacterial_spot": "Peach bacterial",
    "Pepper,_bell___Bacterial_spot": "Pepper bacterial", "Potato___Early_blight": "Potato early blight",
    "Potato___Late_blight": "Potato late blight", "Squash___Powdery_mildew": "Squash powdery",
    "Strawberry___Leaf_scorch": "Strawberry leaf scorch", "Tomato___Bacterial_spot": "Tomato bacterial",
    "Tomato___Early_blight": "Tomato early blight", "Tomato___Late_blight": "Tomato late blight",
    "Tomato___Leaf_Mold": "leaf mold", "Tomato___Septoria_leaf_spot": "Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider mites", "Tomato___Target_Spot": "target spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow leaf curl", "Tomato___Tomato_mosaic_virus": "mosaic",
}


def main() -> None:
    assert len(NAMES) == 38
    used = []
    for name in NAMES:
        info = get_info(name)
        assert info["known"], f"no KB entry for {name}"
        assert info["severity"] in SEVERITY_ORDER, name
        if name.endswith("healthy"):
            assert info["healthy"] and info["type"] == "Healthy" and info["prevention"], name
        else:
            assert not info["healthy"], name
            assert EXPECTED[name].lower() in info["name"].lower(), (name, info["name"])
            for key in ("summary", "symptoms", "immediate", "organic", "chemical", "prevention"):
                assert info[key], f"{name}: empty {key}"
            used.append(info["name"])
    assert len(used) == len(set(used)) == 26, f"expected 26 distinct disease entries, got {len(set(used))}"
    assert len(_DISEASES) == 26
    # all lowercase / odd naming variants still resolve
    assert get_info("Tomato_Early_blight")["name"] == "Tomato early blight"
    assert get_info("tomato___late_blight")["name"] == "Tomato late blight"
    assert get_info("Mango___Anthracnose")["known"] is False          # unknown class -> generic fallback
    print("Knowledge-base tests passed: 38 classes, 26 disease entries, 12 healthy.")


if __name__ == "__main__":
    main()
