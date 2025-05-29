import pytest
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global fixtures
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    # Mock environment variables
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")
    
    # Create temp files
    (tmp_path / "mock_sam.pth").touch()
    (tmp_path / "mock_model.h5").touch()
    (tmp_path / "mock_gardiner.xlsx").touch()
    
    # Change working directory
    monkeypatch.chdir(tmp_path)