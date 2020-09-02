import torch
import torch.nn as nn
import torch.nn.functional as func
from  abstract_layers import DetectionLayer, EmptyLayer
import numpy as np

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

    print (prev_filters, filters)
    # Add convolutional layer to the layer sequence
    conv_sublayer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = not bool(batch_normalize))
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
        end = int(layer_data["layers"][1]) # Index end to 0 if no route end is given
    except IndexError:
        bEnd = False

    # Use empty layer
    route = EmptyLayer()
    module.add_module("route_{0}".format(layer_index), route)

    # Set the feature maps of this layer to the concatenation of the end indexed feature maps and
    # start indexed feature maps if the end index is specified
    if bEnd and end < layer_index:
        filters = output_filters[start] + output_filters[end]
    else:
        filters = output_filters[start]

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
    anchors = anchors[0:len(mask) - 1]

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
            route_layer(module, layer_index, layer_data, output_filters, filters)
        elif layer_type == 'shortcut':
            shortcut_layer(module, layer_index)
        elif layer_type == 'yolo':
            yolo_layer(module, layer_index, layer_data)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)


    return (cnn_metadata, module_list)
