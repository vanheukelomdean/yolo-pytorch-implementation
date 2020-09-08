import argparse

def parse_cfg(cfg_file_path: str):
    """
        Parses config file into a list of darknet-layer metadata

    :return: blocks - List of darknet-layers dicts containing layer metadata
    """
    cfg_file = open(cfg_file_path, 'r')
    file_lines = cfg_file.read().split('\n')

    # Drop empty string lines and comment lines; trim end-space
    file_lines = [x.rstrip().lstrip() for x in file_lines if len(x) > 0 and x[0] != '#']

    # Initalize variables to store model layer data
    block = {}
    blocks = []
    for line in file_lines:
        if line[0] == '[':
            if bool(block):
                # Add block to list and empty block
                blocks.append(block)
                block = {}
            # Set block (darknet-layer) type
            block["type"] = line[1:-1].rstrip()
        else:
            # Trim left and right of the equals sign relative to text position
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    return blocks

def parse_arg():
    """
        Parses cli arguments for running detection

    :return: Dict of argument values
    """
    parser = argparse.ArgumentParser(description='YOLOv3 PyTorch Implementation')

    parser.add_argument("--load", dest='load', help=
                            "Directory to load images for detection",
                            default="images/input/", type=str)
    parser.add_argument("--save", dest = 'save', help =
                        "Directory to save images for detectio",
                        default = "images/output", type = str)

    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--conf", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--iou", dest = "iou_threshold", help = "IOU Overlap threshold for predictions", default = 0.4)
    parser.add_argument("--res", dest = 'resolution', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)

    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Location of config file",
                        default = "model_cfg/yolov3.cfg", type = str)
    parser.add_argument("--wts", dest = 'wtsfile', help =
                        "Location of weights file",
                        default = "trained_weights/yolov3.weights", type = str)
    parser.add_argument("--video", dest = "video", help ="Location of video",
                            default="", type=str)

    return parser.parse_args()

def load_classes(classes_file_path: str):
    """
        Loads class for detection

    :param classes_file_path: Path to class file
    :return:
    """
    fp = open(classes_file_path, "r")
    class_names = fp.read().split("\n")[:-1]
    return class_names
