from util import get_test_input
from darknet_model import DarkNet
import torch

CONFIG_FILE_PATH = "./model_cfg/yolov3.cfg"
WEIGHTS_FILE_PATH = "./trained_weights/yolov3.weights"

if __name__ == '__main__':
    model = DarkNet(CONFIG_FILE_PATH, torch.cuda.is_available())
    model.load_weights(WEIGHTS_FILE_PATH)

    pred = model(get_test_input())
    print(pred)