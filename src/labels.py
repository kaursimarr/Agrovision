"""Turn PlantVillage folder names into human-readable labels (no TensorFlow needed)."""
from __future__ import annotations


def parse_label(raw: str) -> tuple[str, str, bool]:
    """Split a class folder name into (plant, condition, is_healthy).

    Handles the naming variants found in the common PlantVillage releases:
    ``Tomato___Early_blight``, ``Tomato__Target_Spot``, ``Tomato_Early_blight``,
    ``Pepper__bell___Bacterial_spot``, ``Corn_(maize)___Common_rust_``.
    """
    plant, condition = raw, ""
    for sep in ("___", "__", "_"):
        if sep in raw:
            plant, condition = raw.split(sep, 1)
            break
    plant = " ".join(plant.replace("_", " ").split())
    condition = " ".join(condition.replace("_", " ").split())
    if not condition:
        condition = "Unknown"
    condition = condition[0].upper() + condition[1:]
    return plant, condition, "healthy" in condition.lower()


def pretty_label(raw: str) -> str:
    plant, condition, _ = parse_label(raw)
    return f"{plant} - {condition}"
