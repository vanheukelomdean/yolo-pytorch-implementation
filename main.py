from cfg_parser import parse_cfg, print_cfg

CONFIG_FILE_PATH = "./model_cfg/yolov3.cfg"

if __name__ == '__main__':
    print_cfg(parse_cfg(CONFIG_FILE_PATH))


