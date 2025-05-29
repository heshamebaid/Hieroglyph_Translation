#!/usr/bin/env python3
"""
FastAPI server for hieroglyph translation pipeline:
Image → Symbol Detection → Classification → Translation/Story
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import shutil
import uuid
import json

# Import pipeline classes
from hieroglyph_pipeline import HieroglyphPipeline, HieroglyphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hieroglyph Translation API",
    description="API for translating hieroglyph images to stories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
config = None

# Create base directories
BASE_DIR = Path("output")
IMAGES_DIR = BASE_DIR / "images"
SYMBOLS_DIR = BASE_DIR / "symbols"
TRANSLATIONS_DIR = BASE_DIR / "translations"
JSON_DIR = BASE_DIR / "json"

for dir in [BASE_DIR, IMAGES_DIR, SYMBOLS_DIR, TRANSLATIONS_DIR, JSON_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

def create_session_directory() -> tuple[Path, dict[str, Path]]:
    """Create a unique session directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    session_name = f"session_{timestamp}_{session_id}"
    
    # Create session directories
    dirs = {
        "session": BASE_DIR / session_name,
        "images": IMAGES_DIR / session_name,
        "symbols": SYMBOLS_DIR / session_name,
        "translations": TRANSLATIONS_DIR / session_name,
        "json": JSON_DIR / session_name
    }
    
    for dir in dirs.values():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dirs["session"], dirs

def save_results_to_disk(results: dict, session_dirs: dict[str, Path], filename: str):
    """Save processing results to appropriate directories"""
    base_name = Path(filename).stem
    
    # Save JSON results
    json_path = session_dirs["json"] / f"{base_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save translation/story if present
    if results.get('story'):
        translation_path = session_dirs["translations"] / f"{base_name}_translation.txt"
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(results['story'])
    
    # Update results with file paths
    results['file_paths'] = {
        'json_results': str(json_path),
        'translation': str(translation_path) if results.get('story') else None,
        'symbols_dir': str(session_dirs["symbols"]),
        'input_image': str(session_dirs["images"] / filename)
    }

@app.on_event("startup")
async def startup():
    """Initialize the pipeline on startup"""
    global pipeline, config
    try:
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize configuration
        config = HieroglyphConfig()
        
        # Check for required model files
        if not os.path.exists(config.SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM model not found at {config.SAM_CHECKPOINT_PATH}")
        if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(f"Classifier model not found at {config.CLASSIFIER_MODEL_PATH}")
        
        # Set API key directly
        api_key = "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"
        
        # Initialize pipeline
        pipeline = HieroglyphPipeline(config, api_key=api_key)
        pipeline.setup()
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hieroglyph Translation API",
        "status": "running",
        "endpoints": {
            "/translate": "Translate a hieroglyph image",
            "/health": "Check API health",
            "/config": "Get current configuration"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        return {"status": "error", "message": "Pipeline not initialized"}
    return {"status": "healthy", "message": "Pipeline ready"}

@app.post("/translate")
async def translate_hieroglyph(file: UploadFile = File(...)):
    """Translate a hieroglyph image to text"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create session directories
        session_dir, session_dirs = create_session_directory()
        logger.info(f"Created session directory: {session_dir}")
        
        # Save uploaded file
        input_path = session_dirs["images"] / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        results = pipeline.process_image(str(input_path), str(session_dirs["symbols"]))
        
        # Save results to disk
        save_results_to_disk(results, session_dirs, file.filename)
        
        # Add session information to results
        results["session_dir"] = str(session_dir)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current pipeline configuration"""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not available")
    
    return {
        "image_size": config.IMAGE_SIZE,
        "sam_model_type": config.SAM_MODEL_TYPE,
        "min_mask_area": config.MIN_MASK_AREA,
        "story_enabled": config.STORY_ENABLED,
        "llm_model": config.LLM_MODEL if hasattr(config, 'LLM_MODEL') else None,
        "output_directories": {
            "base": str(BASE_DIR),
            "images": str(IMAGES_DIR),
            "symbols": str(SYMBOLS_DIR),
            "translations": str(TRANSLATIONS_DIR),
            "json": str(JSON_DIR)
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )