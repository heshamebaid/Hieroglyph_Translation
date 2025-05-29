import pytest
from config_loader import load_config

def test_config_loading(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write("IMAGE_SIZE: [300, 300]\nSAM_POINTS_PER_SIDE: 32")
    
    config = load_config(str(config_path))
    assert config.IMAGE_SIZE == [300, 300]
    assert config.SAM_POINTS_PER_SIDE == 32