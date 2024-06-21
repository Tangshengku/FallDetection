# FallDetection
Demo for Fall Detection

# Quickstart
Installation
```bash
pip install -r requirements.txt
```
Run 
```bash
# Make sure your data is prepared as .csv and the last column is the label.
# The path can be the .csv file or a directory with csv files. The program will merge all .csv files automatically.
python train.py --data_path /your/data/path
```

The code was only validated on CPU with less than 10-minute training.