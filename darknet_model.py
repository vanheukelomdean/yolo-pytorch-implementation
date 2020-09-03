import torch
import torch.nn as nn
import torch.nn.functional as func
from abstract_layers import DetectionLayer, EmptyLayer
from cfg_parser import parse_cfg, print_cfg
from util import *

class DarkNet (nn.Module):
    """"
        Model class to override Module forward function for forward passing through cnn layers
    """
    def __init__(self, cfg_file_path: str, cuda: bool = False):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfg_file_path)
        self.cnn_metadata, self.module_list = create_modules(self.blocks)
        self.cuda = cuda

    def forward(self, x):
        # Exclude first 'net' block when iterating
        modules = self.blocks[1:]
        # Cached feature maps of every layer
        outputs = {}
        write = False
        for index, module in enumerate(modules):
            module_type = (module["type"])
            x0 = x
            # Input feature map of next module is the output of the current layer if conv or upsample
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)

            # Set inputs of next layer to previously cached outputs
            elif module_type == "route":
                route_layers = [int(a) for a in module["layers"]]

                # Reset start parameter to num layers behind index
                if (route_layers[0]) > 0:
                    route_layers[0] -= index
                # Feed output of `start` layers behind current layer to route output
                if len(route_layers) == 1:
                    x = outputs[index + route_layers[0]]

                # Start and end route parameters
                else:
                    # Reset end parameter to num layers behind index
                    if (route_layers[1]) > 0:
                        route_layers[1] -= index
                    # Feed concatenated output of `start` layers and `end` layers behind current layer to route output
                    x = torch.cat((outputs[index + route_layers[0]],
                                   outputs[index + route_layers[1]]), 1)

            # Pass inputs to outputs
            elif module_type == "shortcut":
                x = outputs[index - 1] + outputs[index + int(module["from"])]

            elif module_type == "yolo":
                x = predict_transform(x.data,
                                      int (self.cnn_metadata.get("height")),
                                      self.module_list[index][0].anchors,
                                      int (module["classes"]),
                                      self.cuda)
                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)
            # Cache Feature Map
            outputs[index] = x

        return detections

def conv_layer(module, layer_index: int, layer_data: dict, prev_filters: int):
    """
        Adds a convolutional layer to the module layer sequence

    :param module: The module storing the list of sub-module layers
    :param layer_index: Number to name network layers in order
    :param layer_data: Dictionary of parameters to pass in layer constructor
    :param prev_filters: Number of channels feeding into this layer
    """
    activation = layer_data.get('activation')
    batch_normalize = int(layer_data.get('batch_normalize') or 0)
    filters = int(layer_data.get("filters"))
    padding = int(layer_data.get("pad"))
    kernel_size = int(layer_data.get("size"))
    stride = int(layer_data.get("stride"))

    if padding:
        pad = (kernel_size - 1) // 2
    else:
        pad = 0
    # Add convolutional layer to the layer sequence
    conv_sublayer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = not bool(batch_normalize))
    print( layer_index, conv_sublayer)
    module.add_module("conv_{0}".format(layer_index), conv_sublayer)

    # Chain batch normalization layer to convolutional layer
    if batch_normalize:
        batch_norm_sublayer = nn.BatchNorm2d(filters)
        module.add_module("batch_norm_{0}".format(layer_index), batch_norm_sublayer)

    # Chain activation layer to previous layer to introduce non linearity in the model
    # Darknet only uses leaky ReLU activation function to convoutional layers
    if activation == "leaky":
        activation_sublayer = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leaky_{0}".format(layer_index), activation_sublayer)
    return filters

def upsample_layer(module, layer_index: int, layer_data: dict):
    """
        Adds an bilinear-upsampling layer to the network layer sequence

    :param module: The module storing the list of sub-module layers
    :param layer_index: Number to name network layers in order
    :param layer_data:
    """
    stride = int(layer_data["stride"])
    # Apply bilinear upsampling technique
    upsample = nn.Upsample(scale_factor=2, mode="bilinear")
    module.add_module("upsample_{}".format(layer_index), upsample)

def route_layer(module, layer_index: int, layer_data: dict, output_filters, filters):
    """
        Adds a route layer to the network layer sequence

    :param module: The module storing the list of sub-module layers
    :param layer_index: Number to name network layers in order
    :param layer_data: Dictionary of parameters to pass in layer constructor
    :param output_filters: list of computed feature maps from previous layers
    :param filters: feature maps from this layers
    """
    bEnd = True
    # Transform layer index(s) from string to list
    layer_data["layers"] = layer_data["layers"].split(',')

    start = int(layer_data["layers"][0])
    try:
        end = int(layer_data["layers"][1])
    except IndexError:
        bEnd = False

    # Transform parameters into a negative int of num layers back to retrieve feature map
    if start > 0:
        start -= layer_index
    if bEnd and end > 0:
        end -= layer_index

    # Use empty layer
    route = EmptyLayer()
    module.add_module("route_{0}".format(layer_index), route)

    if bEnd and end < 0:
        return output_filters[layer_index + start] + output_filters[layer_index + end]
    else:
        return output_filters[layer_index + start]


def shortcut_layer(module, layer_index: int):
    '''
        Adds a shortcut layer to the network layer sequence

    :param module: The module storing the list of sub-module layers
    :param layer_index: Number to name network layers in order
    '''
    module.add_module("shortcut_{}".format(layer_index), EmptyLayer())

def yolo_layer(module, layer_index: int, layer_data: dict):
    '''
        Adds a final yolo detection layer to the network layer sequence

    :param module: The module storing the list of sub-module layers
    :param layer_index: Number to name network layers in order
    '''
    # Transform the comma separated string into a list of ints
    mask = layer_data["mask"].split(",")
    mask = [int(mask_value) for mask_value in mask]

    # Split, cast, pair coordinates, and cut indices to mask array length
    anchors = layer_data["anchors"].split(",")
    anchors = [int(anchor) for anchor in anchors]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = anchors[0:len(mask)]

    module.add_module("Detection_{}".format(layer_index), DetectionLayer(anchors))

def create_modules(blocks):
    cnn_metadata = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    # defined for back propogation
    filters = None
    output_filters = []
    layer_index = 0
    layer_data ={}
    module = None

    for layer_index, layer_data in enumerate(blocks[1:]):
        module = nn.Sequential()
        layer_type = layer_data.get("type")

        if layer_type == 'convolutional':
            filters = conv_layer(module, layer_index, layer_data, prev_filters)
        elif layer_type == 'upsample':
            upsample_layer(module, layer_index, layer_data)
        elif layer_type == 'route':
            filters = route_layer(module, layer_index, layer_data, output_filters, filters)
        elif layer_type == 'shortcut':
            shortcut_layer(module, layer_index)
        elif layer_type == 'yolo':
            yolo_layer(module, layer_index, layer_data)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (cnn_metadata, module_list)
