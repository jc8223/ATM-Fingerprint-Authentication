# FingerprintMatcher
A Python OpenCV2 project made for Image Processing CSE3020

## Setup
- Install packages with req file or maunually install
```
pip install -r requirements.txt
```
- Open app.py and set `REAL_FINGERPRINT_DIRECTORY` and `scanned_fingerprint_filename`

Dataset sourced from https://www.kaggle.com/datasets/ruizgara/socofing

- Run app.py
```
python app.py
```
## Graph of Threshold Vs Score
- Set `true_fingerprint_name` to the name of file we expect the given fingerprint to correctly match
- Run app.py and choose the "generate csv" option
- A file `results.csv` should be created in the same directory
- Run graph.py
```
python graph.py
```