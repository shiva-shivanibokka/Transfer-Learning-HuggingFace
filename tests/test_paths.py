from src.utils import paths
from configs.vision_config import VISION_MODELS
from configs.text_config import TEXT_MODELS


def test_vision_checkpoint_path_matches_run_dir():
    p = paths.vision_checkpoint_path("dinov2_base", "full_finetune", 1.0)
    assert p.parent == paths.vision_run_dir("dinov2_base", "full_finetune", 1.0)
    assert p.name == "best_model.pt"
    assert p.parts[-2] == "frac1.00"


def test_text_paths():
    assert paths.text_checkpoint_path("roberta").name == "best_model.pt"
    assert paths.text_temperature_path("roberta").name == "temperature.json"


def test_app_vision_model_keys_are_real_registry_keys():
    # The app maps display names -> registry keys. Every mapped key must exist
    # in the config registry, or build_model() will KeyError. (Bug A2 guard.)
    from app.gradio_app import VISION_MODEL_IDS

    for display, key in VISION_MODEL_IDS.items():
        assert key in VISION_MODELS, f"{display} -> {key} not in VISION_MODELS"


def test_app_text_model_keys_are_real_registry_keys():
    from app.gradio_app import TEXT_MODEL_IDS

    for display, key in TEXT_MODEL_IDS.items():
        assert key in TEXT_MODELS, f"{display} -> {key} not in TEXT_MODELS"
