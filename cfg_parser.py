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

def print_cfg(blocks):
    for block in blocks:
        for (k, v) in block.items():
            print(k + ' = ' + v)