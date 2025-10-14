from setuptools import setup, find_packages

setup(
    name="ironman_interface_detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=1.7.0",
        "torchvision>=0.8.1", 
        "opencv-python>=4.5.0",
        "speechrecognition>=3.8.1",
        "PyAudio>=0.2.11",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "fuzzywuzzy>=0.18.0",
        "python-levenshtein>=0.12.0"
    ],
)