from setuptools import setup, find_packages

setup(
    name="hieroglyph_processor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "opencv-python-headless",
        "numpy",
        "torch",
        "tensorflow",
        "scikit-learn",
        "segment-anything",
        "pandas",
        "pyyaml",
        "requests",
        "tqdm",
        "openai"
    ],
    entry_points={
        'console_scripts': [
            'hieroglyph-api = api_server:main',
            'hieroglyph-batch = batch_processor:main',
            'download-models = model_downloader:main'
        ]
    }
)