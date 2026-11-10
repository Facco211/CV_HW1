## Requirements
Python 3.12

## Dependencies
numpy, opencv-python, scikit-image

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# How to run
python script.py

This command runs the complete pipeline, it reads the original images, creates the degraded versions, applies all methods, saves every result and writes the metrics table. 
In order to print the images during the execution of the pipeline set SHOWME = True.

# File structure

-> HW1/
    -> README.md
    -> requirements.txt
    -> script.py
    -> images/
        -> image1.jpg
        ...
    -> results/
        -> metrics.csv
        -> image1/
            -> degraded.png
            -> stretch.png
            -> equalize.png
            -> nlm_stretch.png
            -> comparison.png
        -> image2/
            ...
        ...

! The results folder is created after the execution of script.py
