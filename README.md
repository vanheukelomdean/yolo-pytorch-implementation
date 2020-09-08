# YOLO PyTorch Implementation

This project is a small pytorch implementation of a YOLOv3 model with darknet architecture from a config file. The intent of this project is mainly educational and closely follows this [tutorial](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch).

## Installation
  - Clone this repository
  - Install dependencies using `pip install -r requirements.txt`

## Usage
Run module `detect.py` with any of the below options:

- `--load`: str - Image load directory   
- `--save`: str - Image save directory  
- `--cfg`: str - Config file location  
- `--wts`: str - Trained model weights file location 
- `--bs`: int - Batch size    
- `--res`: int Network input resolution
- `--conf`: float (0 < x < 1) - Min objectness confidence   
- `--iou`: float (0 < x < 1) - Max IoU for diffetent object instances 



## Authors
Dean Van Heukelom - [vanheukelomdean](https://github.com/vanheukelomdean)