from util import get_test_input
from darknet_model import DarkNet
import torch

CONFIG_FILE_PATH = "./model_cfg/yolov3.cfg"

if __name__ == '__main__':
    model = DarkNet(CONFIG_FILE_PATH, torch.cuda.is_available())
    print("Model Created")

    pred = model(get_test_input())
    print(pred)