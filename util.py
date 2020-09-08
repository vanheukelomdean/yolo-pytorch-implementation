from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
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

def write_results(prediction, objectness_threshold, num_classes, overlap_threshold = 0.4):
    """
        Determines the most accurate detection and class identified by the detections

    :param prediction: Tensor of prediction data for all images in batch and all potential detections per image
    :param objectness_threshold: Required objectness confidence for a detection
    :param num_classes: Number of object classifications
    :param overlap_threshold: Required iou overlap for a detection
    :return: Tensor of length 8: batch image index, corner coordinates, objectness confidence, class score and index
    """
    # Filter all predictions with objectness greater than the freeshold (remove all less than)
    conf_mask = (prediction[:,:,4] > objectness_threshold).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # Transform center (x,y) , (w, h) -> (l, t), (r, b)
    ltrb_box = prediction.new(prediction.shape)
    # Left and top
    ltrb_box[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    ltrb_box[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    # Right and bottom
    ltrb_box[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    ltrb_box[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = ltrb_box[:, :, :4]
    # Must loop over the images in the batch because the dynamic amount of detctions per image
    # can not be vectorized
    batch_size = prediction.size(0)
    write = False
    for i in range(batch_size):
        image_prediction = prediction[i]
        # Get the max confidence index and score for each detction in the images prediction list
        max_index, max_score = torch.max(image_prediction[:, 5:5 + num_classes], 1)
        # Transpose tensors
        max_index = max_index.float().unsqueeze(1)
        max_score = max_score.float().unsqueeze(1)
        # Concatenate max vectors to image predictions
        image_pred = torch.cat((image_prediction[:, :5], max_index, max_score), 1)
        # Remove zeroed rows
        non_zero_conf = torch.nonzero(image_pred[:, 4])
        # Catch images with no detections when redimensioning prediction list
        try:
            image_pred_ = image_pred[non_zero_conf.squeeze(),:].view(-1,7)
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue
        # Get a set of classes in the image
        image_classes = distinct_tensor(image_pred_[:, -1])
        for class_ in image_classes:
            image_pred_class = non_max_suppression(image_pred_, class_, overlap_threshold)
            batch_index = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)
            # Only create the ouput tensor if there are any detections
            if not write:
                output = torch.cat((batch_index, image_pred_class),1)
                write = True
            # Append next detections to output tensor
            else:
                out = torch.cat((batch_index, image_pred_class),1)
                output = torch.cat((output,out))
    # Safely return output in case of no detections
    try:
        return output
    except:
        return 0

def distinct_tensor(tensor):
    """
        Returns a new tensor containing all distinct values of the param tensor

    :param tensor: Tensor to detect distinct values in
    :return: Distinct valued Tensor
    """
    # Filter non distict values
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    # Copy tensor and return
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def non_max_suppression(prediction, class_, overlap_threshold):
    """
        Filters and returns detections of a single class by largest IOU

    :param prediction:
    :param class_:
    :param overlap_threshold:
    :return:
    """
    # Filter predictions of a single class
    class_mask = prediction * (prediction[:, -1] == class_).float().unsqueeze(1)
    class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()
    prediction_class = prediction[class_mask_index].view(-1, 7)
    # Sort confideces of the class descending
    conf_sort_index = torch.sort(prediction_class[:, 4], descending=True)[1]
    prediction_class = prediction_class[conf_sort_index]
    num_detections = prediction_class.size(0)

    for i in range(num_detections):
        try:
            # Get intersection over union of all detections for this class
            ious = bbox_iou(prediction_class[i].unsqueeze(0), prediction_class[i+1:])
        except ValueError:
            break
        except IndexError:
            break
        # Filter all detections with iou less than the threshold
        # print(ious, overlap_threshold)
        iou_mask = (ious < overlap_threshold).float().unsqueeze(1)
        prediction_class[i+1:] *= iou_mask
        non_zero_ind = torch.nonzero(prediction_class[:,4]).squeeze()
        prediction_class = prediction_class[non_zero_ind].view(-1,7)
    return prediction_class

def bbox_iou(bbox1, bbox2):
    """
        Calculates and returns intersection over union metric

    :param bbox1: Box 1
    :param bbox2: Box 2
    :return: ratio area of intersection to area of union
    """

    # Get the coordinates of bounding boxes
    left1, top1, right1, bottom1 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    left2, top2, right2, bottom2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # Get the coordinates of the intersection rectangle
    left_inter = torch.max(left1, left2)
    top_inter = torch.max(top1, top2)
    right_inter = torch.min(right1, right2)
    bottom_inter = torch.min(bottom1, bottom2)

    # Intersection area
    inter_area = torch.clamp(right_inter - left_inter + 1, min=0) * \
                 torch.clamp(bottom_inter - top_inter + 1, min=0)
    # Box areas
    area1 = (right1 - left1 + 1) * (bottom1 - top1 + 1)
    area2 = (right2 - left2 + 1) * (bottom2 - top2 + 1)

    return inter_area / (area1 + area2 - inter_area)


def pad_image(image, input_dimension):
    """
        Resizes image to model input dimensions while maintaining aspect ratio
            by padding image margins witn grey
    :param image: Frame to resize
    :param input_dimension: Dimensions of cnn input variable
    :return:
    """
    image_h, image_w = image.shape[0], image.shape[1]

    width, height = input_dimension
    min_rescale = min(width/image_w, height/image_h)
    new_width, new_height = int(image_w * min_rescale), int(image_h * min_rescale)
    resized_image = cv2.resize(image, (new_width,new_height), interpolation=cv2.INTER_CUBIC)

    # Fill numpy array with grey pixels
    canvas = np.full((width, height, 3), 128)
    # Calculate margins to pad
    horizontal_pad = (width - new_width) // 2
    vertical_pad = (height - new_height) // 2
    #Place image in center
    canvas[vertical_pad:vertical_pad + new_height,
            horizontal_pad:horizontal_pad + new_width, :] = resized_image
    return canvas

def prepare_input_image(image, input_dimension):
    """
        Return image as a variable for cnn input
    :param image: Frame to input
    :param input_dimension: Dimension to resize to
    :return:Input variable
    """
    print(len(image))
    image = pad_image(image, (input_dimension, input_dimension))
    image = image[:,:,::-1].transpose((2,0,1)).copy()
    return torch.from_numpy(image).float().div(255.0).unsqueeze(0)

def get_test_input():
    img = cv2.imread("images/input/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)
    return img_

def draw(x, results, class_, color, video = False):
    if video:
        img = results
    else:
        img = results[int(x[0])]
    # Color the bbox
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    label = "{0}".format(class_)
    cv2.rectangle(img, c1, c2, color, 1)
    # color the label surrounded by rectangle-filled bbox color
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    return img