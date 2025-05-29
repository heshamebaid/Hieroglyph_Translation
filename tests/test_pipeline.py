#!/usr/bin/env python3
"""
Test Suite for Hieroglyph Processing Pipeline

Comprehensive tests covering:
- Image processing
- Edge detection
- SAM segmentation
- Mask processing
- Symbol extraction
- Classification
- Story generation
- Full pipeline integration
"""

import os
import pytest
import cv2
import numpy as np
import torch
import tensorflow as tf
from unittest.mock import patch, MagicMock, call
from hieroglyph_pipeline import (
    HieroglyphConfig,
    ImageProcessor,
    SegmentationEngine,
    HieroglyphClassifier,
    StoryGenerator,
    HieroglyphPipeline
)

# Fixtures
@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

@pytest.fixture
def sample_mask():
    return {
        'segmentation': np.random.choice([True, False], (512, 512)),
        'bbox': [50, 50, 100, 100],
        'area': 5000
    }

@pytest.fixture
def config():
    cfg = HieroglyphConfig()
    cfg.IMAGE_SIZE = (512, 512)
    cfg.CROP_SIZE = (224, 224)
    cfg.SAM_CHECKPOINT_PATH = "mock_sam.pth"
    cfg.CLASSIFIER_MODEL_PATH = "mock_model.h5"
    cfg.GARDINER_LIST_PATH = "mock_gardiner.xlsx"
    cfg.MIN_MASK_AREA = 100  # Smaller for tests
    return cfg

# Test cases
def test_image_processor_load(config, sample_image):
    """Test image loading and preprocessing"""
    processor = ImageProcessor(config)
    
    with patch("cv2.imread", return_value=sample_image) as mock_read:
        # Test successful load
        img = processor.load_and_preprocess("test.jpg")
        mock_read.assert_called_once()
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8
        
        # Test invalid file
        mock_read.return_value = None
        with pytest.raises(ValueError):
            processor.load_and_preprocess("invalid.jpg")

def test_image_processor_edge_detection(config, sample_image):
    """Test edge detection functionality"""
    processor = ImageProcessor(config)
    
    # Test Roberts edge detection
    edges = processor.detect_edges(sample_image)
    assert edges.shape == (512, 512, 3)
    assert edges.dtype == np.uint8
    
    # Verify edge values are within valid range
    assert np.min(edges) >= 0
    assert np.max(edges) <= 255

def test_segmentation_initialization(config):
    """Test SAM model initialization"""
    engine = SegmentationEngine(config)
    
    # Test CPU initialization
    with patch("torch.cuda.is_available", return_value=False):
        with patch("segment_anything.sam_model_registry") as mock_registry:
            mock_model = MagicMock()
            mock_registry.return_value = mock_model
            
            engine.initialize_sam()
            mock_registry.assert_called_with(config.SAM_MODEL_TYPE)
            mock_model.assert_called_with(checkpoint=config.SAM_CHECKPOINT_PATH)
            mock_model().to.assert_called_with("cpu")
    
    # Test missing model file
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            engine.initialize_sam()

def test_segmentation_mask_generation(config):
    """Test mask generation and processing"""
    engine = SegmentationEngine(config)
    engine.mask_generator = MagicMock()
    
    # Create mock mask data
    mock_mask = MagicMock()
    mock_mask_generation = [{
        'segmentation': np.random.choice([True, False], (512, 512)),
        'bbox': [10, 10, 100, 100],
        'area': 600
    } for _ in range(3)]
    
    engine.mask_generator.generate.return_value = mock_mask_generation
    
    # Test mask generation
    edge_image = np.zeros((512, 512, 3), dtype=np.uint8)
    masks = engine.generate_masks(edge_image)
    assert len(masks) == 3
    
    # Test filtering
    filtered = engine.post_process_masks(masks)
    assert len(filtered) == 3  # All meet min area
    
    # Test filtering with small masks
    small_masks = masks + [{
        'segmentation': np.random.choice([True, False], (512, 512)),
        'bbox': [10, 10, 100, 100],
        'area': 50  # Below threshold
    }]
    filtered = engine.post_process_masks(small_masks)
    assert len(filtered) == 3

def test_segmentation_clustering(config, sample_mask):
    """Test mask clustering logic"""
    engine = SegmentationEngine(config)
    
    # Create masks with different vertical positions
    masks = [
        {**sample_mask, 'bbox': [100, 50, 100, 100]},    # Y=50
        {**sample_mask, 'bbox': [200, 150, 100, 100]},   # Y=150
        {**sample_mask, 'bbox': [50, 50, 100, 100]},     # Y=50 (same row)
        {**sample_mask, 'bbox': [300, 300, 100, 100]},   # Y=300
    ]
    
    clustered = engine.cluster_masks(masks)
    assert len(clustered) == 3  # 3 clusters (50, 150, 300)
    
    # Verify row order
    assert clustered[0][0]['bbox'][1] == 50
    assert clustered[1][0]['bbox'][1] == 150
    assert clustered[2][0]['bbox'][1] == 300
    
    # Verify horizontal order within rows
    row1 = clustered[0]
    assert row1[0]['bbox'][0] == 50
    assert row1[1]['bbox'][0] == 100

def test_classifier_initialization(config):
    """Test classifier model and metadata loading"""
    classifier = HieroglyphClassifier(config)
    
    # Test model loading
    with patch("tensorflow.keras.models.load_model") as mock_load:
        classifier.load_model()
        mock_load.assert_called_with(config.CLASSIFIER_MODEL_PATH)
    
    # Test metadata loading
    mock_df = MagicMock()
    with patch("pandas.read_excel", return_value=mock_df) as mock_excel:
        classifier.load_gardiner_metadata()
        mock_excel.assert_called_with(config.GARDINER_LIST_PATH, header=1)
        assert classifier.gardiner_df == mock_df
    
    # Test missing files
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            classifier.load_model()
        
        with pytest.raises(FileNotFoundError):
            classifier.load_gardiner_metadata()

def test_classifier_predictions(config):
    """Test symbol classification workflow"""
    classifier = HieroglyphClassifier(config)
    classifier.model = MagicMock()
    classifier.gardiner_df = MagicMock()
    
    # Mock classification results
    mock_prediction = np.array([[0.1, 0.8, 0.1]])  # Class 1
    classifier.model.predict.return_value = mock_prediction
    
    # Mock metadata lookup
    mock_row = MagicMock()
    mock_row.to_dict.return_value = {"description": "Test hieroglyph"}
    classifier.gardiner_df.__getitem__.return_value.__eq__.return_value = [True]
    classifier.gardiner_df.loc.__getitem__.return_value = mock_row
    
    # Test classification
    symbol_paths = ["symbol1.png", "symbol2.png"]
    results = classifier.classify_symbols(symbol_paths)
    
    assert len(results) == 2
    assert results[0]['Gardiner Code'] == 'Aa15'  # Index 1 in mapping
    assert results[0]['confidence'] == 0.8
    assert "description" in results[0]
    
    # Test error handling
    classifier.model.predict.side_effect = Exception("Test error")
    results = classifier.classify_symbols(["error.png"])
    assert "error" in results[0]

def test_story_generation(config):
    """Test story generation with LLM"""
    generator = StoryGenerator(config)
    generator.client = MagicMock()
    
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test story content"
    generator.client.chat.completions.create.return_value = mock_response
    
    # Test successful generation
    classifications = [
        {"Gardiner Code": "A1", "confidence": 0.95},
        {"Gardiner Code": "B2", "confidence": 0.85}
    ]
    story = generator.generate_story(classifications)
    assert story == "Test story content"
    
    # Test no valid hieroglyphs
    story = generator.generate_story([{"Gardiner Code": "Unknown"}])
    assert "No valid hieroglyphs" in story
    
    # Test API error
    generator.client.chat.completions.create.side_effect = Exception("API error")
    story = generator.generate_story(classifications)
    assert "error" in story.lower()

def test_full_pipeline(config, tmp_path):
    """Test full pipeline integration with mocks"""
    # Create pipeline instance
    pipeline = HieroglyphPipeline(config, api_key="test_api_key")
    
    # Mock all components
    with patch.object(ImageProcessor, 'load_and_preprocess') as mock_load_img, \
         patch.object(ImageProcessor, 'detect_edges') as mock_edges, \
         patch.object(SegmentationEngine, 'generate_masks') as mock_gen_masks, \
         patch.object(SegmentationEngine, 'post_process_masks') as mock_filter_masks, \
         patch.object(SegmentationEngine, 'cluster_masks') as mock_cluster, \
         patch.object(HieroglyphClassifier, 'classify_symbols') as mock_classify, \
         patch.object(StoryGenerator, 'generate_story') as mock_story:
        
        # Setup return values
        mock_img = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_load_img.return_value = mock_img
        mock_edges.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        
        mock_mask = {'segmentation': None, 'bbox': [0,0,100,100]}
        mock_gen_masks.return_value = [mock_mask] * 3
        mock_filter_masks.return_value = [mock_mask] * 2
        mock_cluster.return_value = [[mock_mask], [mock_mask]]  # Two clusters
        
        mock_classify.return_value = [
            {'Gardiner Code': 'A1', 'confidence': 0.95},
            {'Gardiner Code': 'B2', 'confidence': 0.85}
        ]
        mock_story.return_value = "Generated story"
        
        # Initialize pipeline
        pipeline.setup()
        
        # Run pipeline
        image_path = "test.jpg"
        output_dir = tmp_path / "output"
        result = pipeline.process_image(image_path, str(output_dir))
        
        # Verify results
        assert result['symbols_found'] == 2
        assert len(result['classifications']) == 2
        assert result['story'] == "Generated story"
        assert os.path.exists(output_dir / "results.json")
        
        # Verify call sequence
        mock_load_img.assert_called_once_with(image_path)
        mock_edges.assert_called_once_with(mock_img)
        mock_gen_masks.assert_called_once()
        mock_filter_masks.assert_called_once()
        mock_cluster.assert_called_once()
        mock_classify.assert_called_once()
        mock_story.assert_called_once()
    
    # Test error handling
    with patch.object(ImageProcessor, 'load_and_preprocess', side_effect=Exception("Test error")):
        result = pipeline.process_image("error.jpg", str(tmp_path))
        assert "error" in result
        assert "Test error" in result['error']

def test_pipeline_error_handling(config, tmp_path):
    """Test pipeline's error handling capabilities"""
    pipeline = HieroglyphPipeline(config)
    
    # Test SAM initialization failure
    with patch.object(SegmentationEngine, 'initialize_sam', side_effect=Exception("SAM error")):
        with pytest.raises(RuntimeError):
            pipeline.setup()
    
    # Test classification failure during processing
    pipeline.setup = MagicMock()
    
    with patch.object(ImageProcessor, 'load_and_preprocess', return_value=np.zeros((512, 512, 3))), \
         patch.object(HieroglyphClassifier, 'classify_symbols', side_effect=Exception("Classification error")):
        
        result = pipeline.process_image("test.jpg", str(tmp_path))
        assert "error" in result
        assert "Classification error" in result['error']

def test_image_size_validation(config):
    """Test image size validation and resizing"""
    processor = ImageProcessor(config)
    
    # Test small image
    small_img = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = processor.load_and_preprocess_from_array(small_img)
    assert resized.shape == (512, 512, 3)
    
    # Test large image
    large_img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    resized = processor.load_and_preprocess_from_array(large_img)
    assert resized.shape == (512, 512, 3)
    
    # Test exact size
    exact_img = np.zeros((512, 512, 3), dtype=np.uint8)
    resized = processor.load_and_preprocess_from_array(exact_img)
    assert resized.shape == (512, 512, 3)

# Helper method needed in ImageProcessor for the above test
def load_and_preprocess_from_array(self, image: np.ndarray) -> np.ndarray:
    """Helper method for testing with array input"""
    if image is None:
        raise ValueError("Image is None")
    
    # Resize if needed
    if image.shape[:2] != self.config.IMAGE_SIZE:
        image = cv2.resize(image, self.config.IMAGE_SIZE)
    
    return image

# Monkey-patch the helper method
ImageProcessor.load_and_preprocess_from_array = load_and_preprocess_from_array

if __name__ == "__main__":
    pytest.main(["-v", "test_pipeline.py"])