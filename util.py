from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, input_dim, anchors, num_classes, cuda = False):
    """
        This function transforms the prediction coordinates from a reference
            anchor to reference the input image size

    :param prediction: Tensor storing bounding box center coordinates (0,1), width (2), height (3),
                            objectness score (4) and classification scores (5, 5 + number of classes)
    :param input_dim: Input image size
    :param anchors: Coordinates of features
    :param num_classes: Number of classes to identify
    :param cuda: Boolean to use gpu acceleration
    :return prediction: Transfromed tensor to position relative to image size
    """
    batch_size = prediction.size(0)
    # Stride: ratio of input image dimensions to detection map dimensions
    stride = input_dim // prediction.size(2)
    # Dimensions of detection map
    grid_size = input_dim // stride
    # Attributes in bounding box is center x and y, box width and height, objectness score, and each classification score
    bbox_attributes = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attributes * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attributes)

    # Scale anchors down to detection map by factor of stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the bounding box center coordinates
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    # Sigmoid the object confidence
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    # Create matrices of the x cell offset for all columns, and y cell offsets all rows
    a,b = np.meshgrid(grid, grid)
    # Init tensors into single row, many columns
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    # Pass to gpu
    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    # Concatenate offset tensors for all anchors and reshape tensors to two rows
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # Add offsets to all anchor (center coordinates)
    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2: 4] = torch.exp(prediction[:, :, 2: 4]) * anchors
    # Sigmoid the classification scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # Resize bounding box attributes from detection map back to input image size
    prediction[:, :, :4] *= stride

    return prediction

def get_test_input():
    img = cv2.imread("images/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)
    return img_