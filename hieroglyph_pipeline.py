#!/usr/bin/env python3
"""
Hieroglyph Processing Pipeline

Main module for processing hieroglyph images through:
1. Edge detection and segmentation
2. Symbol extraction
3. Classification
4. Story generation
"""

import os
import json
import logging
import time
import numpy as np
import cv2
import pandas as pd
import torch
import tensorflow as tf
from sklearn.cluster import DBSCAN
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log")
    ]
)
logger = logging.getLogger("hieroglyph_pipeline")

class HieroglyphConfig:
    """Configuration for hieroglyph processing pipeline"""
    # Image processing
    IMAGE_SIZE = (512, 512)
    CROP_SIZE = (299, 299)
    GAUSSIAN_KERNEL = (5, 5)
    
    # SAM parameters
    SAM_MODEL_TYPE = "vit_b"
    SAM_POINTS_PER_SIDE = 64
    SAM_PRED_IOU_THRESH = 0.8
    SAM_STABILITY_SCORE_THRESH = 0.85
    SAM_CROP_N_LAYERS = 1
    SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 2
    SAM_MIN_MASK_REGION_AREA = 70
    
    # Post-processing
    MIN_MASK_AREA = 500
    CLUSTERING_EPS = 20
    CLUSTERING_MIN_SAMPLES = 1
    
    # Paths (relative to project root)
    SAM_CHECKPOINT_PATH = "models/sam_vit_b.pth"
    CLASSIFIER_MODEL_PATH = "models/InceptionV3_model.h5"
    GARDINER_LIST_PATH = "data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx"
    
    # Story generation
    STORY_ENABLED = True
    LLM_MODEL = "qwen/qwen3-30b-a3b:free"
    STORY_PROMPT = """As an expert Egyptologist, analyze these hieroglyphs and create a culturally accurate translation or story:

{symbols_info}

Create a detailed translation that:
1. Incorporates the meanings of the detected symbols
2. Follows ancient Egyptian narrative style
3. Considers the confidence levels of the classifications
4. Explains any significant cultural or historical context
5. Notes any interesting patterns or repeated symbols

Focus on symbols with higher confidence scores (>0.5) but consider lower confidence symbols as supporting context.

Translation:"""

class ImageProcessor:
    """Handles image preprocessing and edge detection"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load, validate, and preprocess image"""
        self.logger.info(f"Loading image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unsupported image format: {image_path}")
        
        # Convert to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.config.IMAGE_SIZE)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Roberts Cross edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.config.GAUSSIAN_KERNEL, 0)
        
        # Roberts Cross operators
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # Apply filters
        dx = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
        
        # Combine gradients and convert to 8-bit
        edges = np.hypot(dx, dy)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.stack([edges] * 3, axis=-1)

class SegmentationEngine:
    """Handles SAM-based segmentation"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sam_model = None
        self.mask_generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def initialize_sam(self):
        """Initialize SAM model and mask generator"""
        self.logger.info("Initializing SAM model")
        if not os.path.exists(self.config.SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM checkpoint not found: {self.config.SAM_CHECKPOINT_PATH}")
        
        # Load SAM model
        self.sam_model = sam_model_registry[self.config.SAM_MODEL_TYPE](
            checkpoint=self.config.SAM_CHECKPOINT_PATH
        )
        self.sam_model.to(self.device)
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=self.config.SAM_POINTS_PER_SIDE,
            pred_iou_thresh=self.config.SAM_PRED_IOU_THRESH,
            stability_score_thresh=self.config.SAM_STABILITY_SCORE_THRESH,
            crop_n_layers=self.config.SAM_CROP_N_LAYERS,
            crop_n_points_downscale_factor=self.config.SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
            min_mask_region_area=self.config.SAM_MIN_MASK_REGION_AREA
        )
    
    def generate_masks(self, edge_image: np.ndarray) -> list:
        """Generate segmentation masks from edge image"""
        return self.mask_generator.generate(edge_image)
    
    def post_process_masks(self, masks: list) -> list:
        """Filter masks by minimum area"""
        return [m for m in masks if np.sum(m['segmentation']) > self.config.MIN_MASK_AREA]
    
    def cluster_masks(self, masks: list) -> list:
        """Cluster masks into rows using DBSCAN"""
        if not masks:
            return []
        
        # Extract y-coordinates
        ys = np.array([m['bbox'][1] for m in masks]).reshape(-1, 1)
        
        # Cluster vertically
        clustering = DBSCAN(
            eps=self.config.CLUSTERING_EPS,
            min_samples=self.config.CLUSTERING_MIN_SAMPLES
        ).fit(ys)
        
        # Group by clusters
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(masks[i])
        
        # Sort clusters vertically and masks horizontally
        sorted_clusters = sorted(
            clusters.values(), 
            key=lambda x: np.mean([m['bbox'][1] for m in x])
        )
        return [
            sorted(cluster, key=lambda m: m['bbox'][0])
            for cluster in sorted_clusters
        ]

class HieroglyphClassifier:
    """Classifies hieroglyph symbols"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.gardiner_df = None
        # Create reverse mapping (index -> Gardiner Code) from original mapping
        gardiner_to_index = self._create_class_mapping()
        self.index_to_class = {index: code for code, index in gardiner_to_index.items()}
    
    def _create_class_mapping(self) -> dict:
        """Gardiner code to class index mapping (for model output)"""
        return {
                'A55': 0, 'Aa15': 1, 'Aa26': 2, 'Aa27': 3, 'Aa28': 4, 'D1': 5, 'D10': 6, 'D156': 7, 'D19': 8, 'D2': 9,
                'D21': 10, 'D28': 11, 'D34': 12, 'D35': 13, 'D36': 14, 'D39': 15, 'D4': 16, 'D46': 17, 'D52': 18, 'D53': 19,
                'D54': 20, 'D56': 21, 'D58': 22, 'D60': 23, 'D62': 24, 'E1': 25, 'E17': 26, 'E23': 27, 'E34': 28, 'E9': 29,
                'F12': 30, 'F13': 31, 'F16': 32, 'F18': 33, 'F21': 34, 'F22': 35, 'F23': 36, 'F26': 37, 'F29': 38, 'F30': 39,
                'F31': 40, 'F32': 41, 'F34': 42, 'F35': 43, 'F4': 44, 'F40': 45, 'F9': 46, 'G1': 47, 'G10': 48, 'G14': 49,
                'G17': 50, 'G21': 51, 'G25': 52, 'G26': 53, 'G29': 54, 'G35': 55, 'G36': 56, 'G37': 57, 'G39': 58, 'G4': 59,
                'G40': 60, 'G43': 61, 'G5': 62, 'G50': 63, 'G7': 64, 'H6': 65, 'I10': 66, 'I5': 67, 'I9': 68, 'L1': 69, 'M1': 70,
                'M12': 71, 'M16': 72, 'M17': 73, 'M18': 74, 'M195': 75, 'M20': 76, 'M23': 77, 'M26': 78, 'M29': 79, 'M3': 80,
                'M4': 81, 'M40': 82, 'M41': 83, 'M42': 84, 'M44': 85, 'M8': 86, 'N1': 87, 'N14': 88, 'N16': 89, 'N17': 90,
                'N18': 91, 'N19': 92, 'N2': 93, 'N24': 94, 'N25': 95, 'N26': 96, 'N29': 97, 'N30': 98, 'N31': 99, 'N35': 100,
                'N36': 101, 'N37': 102, 'N41': 103, 'N5': 104, 'O1': 105, 'O11': 106, 'O28': 107, 'O29': 108, 'O31': 109,
                'O34': 110, 'O4': 111, 'O49': 112, 'O50': 113, 'O51': 114, 'P1': 115, 'P13': 116, 'P6': 117, 'P8': 118,
                'P98': 119, 'Q1': 120, 'Q3': 121, 'Q7': 122, 'R4': 123, 'R8': 124, 'S24': 125, 'S28': 126, 'S29': 127,
                'S34': 128, 'S42': 129, 'T14': 130, 'T20': 131, 'T21': 132, 'T22': 133, 'T28': 134, 'T30': 135, 'U1': 136,
                'U15': 137, 'U28': 138, 'U33': 139, 'U35': 140, 'U7': 141, 'V13': 142, 'V16': 143, 'V22': 144, 'V24': 145,
                'V25': 146, 'V28': 147, 'V30': 148, 'V31': 149, 'V4': 150, 'V6': 151, 'V7': 152, 'W11': 153, 'W14': 154,
                'W15': 155, 'W18': 156, 'W19': 157, 'W22': 158, 'W24': 159, 'W25': 160, 'X1': 161, 'X6': 162, 'X8': 163,
                'Y1': 164, 'Y2': 165, 'Y3': 166, 'Y5': 167, 'Z1': 168, 'Z11': 169, 'Z7': 170
                    }
    
    def load_model(self):
        """Load classification model"""
        self.logger.info("Loading classifier model")
        if not os.path.exists(self.config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {self.config.CLASSIFIER_MODEL_PATH}")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        self.model = tf.keras.models.load_model(self.config.CLASSIFIER_MODEL_PATH)
    
    def load_gardiner_metadata(self):
        """Load and clean Gardiner's sign list metadata"""
        self.logger.info("Loading Gardiner metadata")
        if not os.path.exists(self.config.GARDINER_LIST_PATH):
            raise FileNotFoundError(f"Gardiner list not found: {self.config.GARDINER_LIST_PATH}")
        
        self.gardiner_df = pd.read_excel(self.config.GARDINER_LIST_PATH, header=1)
        self.gardiner_df.rename(columns={
            'Gardiner No.': 'Gardiner Code',
            'Hieroglyph': 'Hieroglyph',
            'Description': 'Description',
            'Details': 'Details'
        }, inplace=True)
        
        # Remove unwanted columns
        unwanted_cols = [col for col in self.gardiner_df.columns if 'Unnamed' in col]
        self.gardiner_df.drop(columns=unwanted_cols, errors='ignore', inplace=True)
        
        # Clean Gardiner codes: remove spaces and ensure string type
        self.gardiner_df['Gardiner Code'] = (
            self.gardiner_df['Gardiner Code']
            .astype(str)
            .str.strip()
            .str.replace(' ', '')
        )
    
    def classify_symbols(self, symbol_paths: list) -> list:
        """Classify symbols and return metadata"""
        results = []
        for path in symbol_paths:
            try:
                # Load and preprocess image
                img = load_img(path, target_size=self.config.CROP_SIZE)
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict class
                prediction = self.model.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = float(np.max(prediction))
                gardiner_code = self.index_to_class.get(class_index, "Unknown")
                
                # Retrieve metadata
                metadata = {"Gardiner Code": gardiner_code, "confidence": confidence}
                if self.gardiner_df is not None:
                    row = self.gardiner_df[self.gardiner_df["Gardiner Code"] == gardiner_code]
                    if not row.empty:
                        metadata.update(row.iloc[0].to_dict())
                
                results.append(metadata)
            except Exception as e:
                self.logger.error(f"Error classifying {path}: {str(e)}")
                results.append({"error": str(e), "path": path})
        return results

class StoryGenerator:
    def __init__(self, config: HieroglyphConfig, api_key: str):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.client = None  # Will be initialized later
        
    def initialize_client(self):
        """Initialize the LLM client"""
        if not self.api_key:
            raise ValueError("API key is required for story generation")
            
        # Initialize client for OpenRouter API
        try:
            import openai
            # Configure OpenAI client for OpenRouter
            openai.api_base = "https://openrouter.ai/api/v1"
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            raise ValueError("OpenAI library is required for story generation")
    
    def _create_prompt(self, classifications: list) -> str:
        """Create detailed prompt from classifications"""
        symbols_info = []
        high_confidence_symbols = []
        medium_confidence_symbols = []
        low_confidence_symbols = []
        
        # Group symbols by confidence
        for cls in classifications:
            if 'error' not in cls:
                confidence = cls.get('confidence', 0)
                symbol_info = (
                    f"Symbol: {cls.get('Hieroglyph', '?')} "
                    f"(Gardiner Code: {cls.get('Gardiner Code', 'Unknown')}, "
                    f"Confidence: {confidence:.1%})\n"
                    f"Meaning: {cls.get('Description', 'Unknown')}\n"
                    f"Details: {cls.get('Details', 'No additional details')}\n"
                )
                
                if confidence > 0.7:
                    high_confidence_symbols.append(symbol_info)
                elif confidence > 0.4:
                    medium_confidence_symbols.append(symbol_info)
                else:
                    low_confidence_symbols.append(symbol_info)
        
        # Combine symbols with headers
        if high_confidence_symbols:
            symbols_info.append("High Confidence Symbols (>70%):")
            symbols_info.extend(high_confidence_symbols)
        
        if medium_confidence_symbols:
            symbols_info.append("\nMedium Confidence Symbols (40-70%):")
            symbols_info.extend(medium_confidence_symbols)
        
        if low_confidence_symbols:
            symbols_info.append("\nLow Confidence Symbols (<40%):")
            symbols_info.extend(low_confidence_symbols)
        
        return self.config.STORY_PROMPT.format(symbols_info="\n".join(symbols_info))
    
    def generate_story(self, classifications: list) -> str:
        """Generate story from classified symbols"""
        if not self.client:
            raise RuntimeError("LLM client not initialized")
            
        # Prepare detailed prompt with classifications
        prompt = self._create_prompt(classifications)
        
        try:
            # Generate story using OpenRouter API
            headers = {
                "HTTP-Referer": "https://github.com/hieroglyph-translation",
                "X-Title": "Hieroglyph Translation API"
            }
            
            response = self.client.Completion.create(
                model=self.config.LLM_MODEL,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7,
                headers=headers
            )
            return response.choices[0].text.strip()
        except Exception as e:
            self.logger.error(f"Story generation failed: {str(e)}")
            return f"Translation could not be generated due to an error: {str(e)}"

class HieroglyphPipeline:
    """Orchestrates the complete hieroglyph processing workflow"""
    def __init__(self, config: HieroglyphConfig, api_key: str = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key or os.getenv("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")
        
        # Initialize components
        self.image_processor = ImageProcessor(config)
        self.segmentation = SegmentationEngine(config)
        self.classifier = HieroglyphClassifier(config)
        
        # Initialize story generator only if enabled in config
        self.story_generator = None
        if config.STORY_ENABLED and self.api_key:
            self.story_generator = StoryGenerator(config, self.api_key)
    
    def setup(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing pipeline components")
        
        try:
            # Initialize segmentation model (SAM)
            self.logger.debug("Initializing SAM segmentation model")
            self.segmentation.initialize_sam()
            
            # Load classification model and metadata
            self.logger.debug("Loading classifier model")
            self.classifier.load_model()
            self.classifier.load_gardiner_metadata()
            
            # Initialize story generator if enabled
            if self.story_generator:
                self.logger.debug("Initializing story generator")
                self.story_generator.initialize_client()
                
        except Exception as e:
            self.logger.critical(f"Pipeline initialization failed: {str(e)}")
            raise RuntimeError(f"Pipeline setup error: {e}") from e
    
    def process_image(self, image_path: str, output_dir: str) -> dict:
        """Process a single image through the pipeline"""
        start_time = time.time()
        results = {
            "processing_time": None,
            "image_path": image_path,
            "symbols_found": 0,
            "classifications": [],
            "story": None,
            "output_dir": output_dir,
            "error": None
        }
        
        try:
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            symbols_dir = os.path.join(output_dir, "symbols")
            os.makedirs(symbols_dir, exist_ok=True)
            
            # 1. Image processing
            image = self.image_processor.load_and_preprocess(image_path)
            edges = self.image_processor.detect_edges(image)
            
            # 2. Segmentation
            masks = self.segmentation.generate_masks(edges)
            masks = self.segmentation.post_process_masks(masks)
            clusters = self.segmentation.cluster_masks(masks)
            
            # 3. Symbol extraction
            symbol_paths = []
            for i, cluster in enumerate(clusters):
                for j, mask in enumerate(cluster):
                    x, y, w, h = map(int, mask['bbox'])
                    symbol = image[y:y+h, x:x+w]
                    path = os.path.join(symbols_dir, f"symbol_{i+1}_{j+1}.png")
                    cv2.imwrite(path, cv2.cvtColor(symbol, cv2.COLOR_RGB2BGR))
                    symbol_paths.append(path)
            
            results["symbols_found"] = len(symbol_paths)
            
            # 4. Classification
            if symbol_paths:
                results["classifications"] = self.classifier.classify_symbols(symbol_paths)
            
            # 5. Story generation (if enabled)
            if self.story_generator and results["classifications"]:
                try:
                    results["story"] = self.story_generator.generate_story(results["classifications"])
                    if not results["story"]:
                        results["error"] = "Story generation produced no output"
                except Exception as e:
                    self.logger.error(f"Story generation failed: {str(e)}")
                    results["error"] = f"Story generation error: {str(e)}"
            elif not self.story_generator:
                results["error"] = "Story generation is not enabled"
            
            # Final processing time
            results["processing_time"] = time.time() - start_time
            
            # Save results
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
                
            return results
        
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results