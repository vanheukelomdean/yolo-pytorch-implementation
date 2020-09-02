from cfg_parser import parse_cfg, print_cfg
from darknet_model import create_modules

CONFIG_FILE_PATH = "./model_cfg/yolov3.cfg"

if __name__ == '__main__':
    blocks = parse_cfg(CONFIG_FILE_PATH)
    print(create_modules(blocks))