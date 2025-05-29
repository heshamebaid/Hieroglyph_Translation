# Hieroglyph Translation API

An advanced API service that translates ancient Egyptian hieroglyphs into meaningful stories using computer vision and AI. The system processes hieroglyph images through a multi-stage pipeline that includes symbol detection, classification, and story generation.

## Features

- 🖼️ **Image Processing**: Advanced edge detection and preprocessing
- 🎯 **Symbol Detection**: Uses Meta's Segment Anything Model (SAM) for accurate hieroglyph segmentation
- 🔍 **Symbol Classification**: Classifies hieroglyphs using a trained InceptionV3 model with Gardiner's Sign List
- 📝 **Story Generation**: Generates culturally accurate translations using advanced LLM (Qwen)
- 🗃️ **Organized Output**: Structured storage of images, symbols, translations, and JSON results
- 🔄 **RESTful API**: Easy-to-use FastAPI endpoints

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for SAM model)
- OpenRouter API key for story generation

## Project Structure

```
hieroglyph-pipeline/
├── hieroglyph_pipeline.py     # Main pipeline code
├── api_server.py              # FastAPI REST server
├── batch_processor.py         # Batch processing script
├── model_downloader.py        # Model download utility
├── config_loader.py           # Configuration management
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── deploy.sh                  # Deployment script
├── config.yaml                # Default configuration
├── README.md                  # This file
├── tests/                     # Test suite
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── test_utils.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                    # Model files
│   ├── sam_vit_b.pth         # SAM model weights
│   └── InceptionV3_model.h5  # Classification model
├── data/                      # Data files
│   └── Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx
├── output/                    # Processing results
├── logs/                      # Log files
└── temp_uploads/              # Temporary files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/hieroglyph-translation.git
cd hieroglyph-translation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required model files:
- Place the SAM model checkpoint (`sam_vit_b.pth`) in `models/`
- Place the classifier model (`InceptionV3_model.h5`) in `models/`
- Place Gardiner's List Excel file in `data/`

## Configuration

The system uses several configuration parameters that can be modified in `hieroglyph_pipeline.py`:

- Image processing parameters (size, kernel)
- SAM model parameters (IoU threshold, stability score)
- Story generation settings (prompt template, model selection)

## API Usage

### Start the Server

```bash
python api_server.py
```

The server runs on `http://localhost:8000` by default.

### API Endpoints

1. **Health Check**
```http
GET /health
```

2. **Get Configuration**
```http
GET /config
```

3. **Translate Hieroglyph Image**
```http
POST /translate
Content-Type: multipart/form-data
file: <image_file>
```

### Example Response

```json
{
  "processing_time": 2.5,
  "image_path": "path/to/input.jpg",
  "symbols_found": 12,
  "classifications": [
    {
      "Gardiner Code": "D21",
      "confidence": 0.85,
      "Description": "Mouth",
      "Details": "Represents speech or eating"
    }
  ],
  "story": "Translation and cultural context...",
  "session_dir": "output/session_20250529_141311_34c2135f",
  "file_paths": {
    "json_results": "path/to/results.json",
    "translation": "path/to/translation.txt",
    "symbols_dir": "path/to/symbols",
    "input_image": "path/to/input.jpg"
  }
}
```

## Output Directory Structure

```
output/
├── images/           # Original uploaded images
├── symbols/         # Extracted hieroglyph symbols
├── translations/    # Generated stories/translations
└── json/           # Complete processing results
```

## Error Handling

The API provides detailed error messages for:
- Invalid image formats
- Missing model files
- Processing failures
- Story generation issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta's Segment Anything Model (SAM)
- Alan Gardiner's Sign List
- OpenRouter/Qwen for LLM capabilities
- TensorFlow and PyTorch communities
