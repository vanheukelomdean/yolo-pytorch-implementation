import os
import os.path as osp
import time
import torch
import pandas as pd
from torch.autograd import Variable
import cv2
import pickle as pkl

from darknet_model import DarkNet
from util import prepare_input_image, write_results, draw
from parse import parse_arg, load_classes

COCO_CLASSES = "classes/coco.names"
COLOR_PALLETE = "classes/pallete"

cuda = torch.cuda.is_available()
args = parse_arg()

confidence = float(args.confidence)
overlap_threshold = float(args.iou_threshold)

classes = load_classes(COCO_CLASSES)
colors = pkl.load(open(COLOR_PALLETE, "rb"))

try:
    model = DarkNet(args.cfgfile, cuda)
    print("DarkNet model loaded")
except FileNotFoundError:
    print ("No file or directory with the name {}".format(args.cfgfile))
    exit()

try:
    model.load_weights(args.wtsfile)
    print ("Weights loaded")
except FileNotFoundError:
    print("No file or directory with the name {}".format(args.wtsfile))
    exit()
input_dimension = int(model.cnn_metadata["height"])
assert input_dimension % 32 == 0
assert input_dimension > 32

if cuda:
    model.cuda()
model.eval()

video_file = None
try:
    """
                                        Video Object Detection
    """
    print (args.video)
    video_file = str(args.video)
except:
    pass

if bool(video_file):
    try:
        capture = cv2.VideoCapture(video_file)
    except:
        print ("Video could not be found")
        exit()

    frames = 0
    start = time.time()

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            img = prepare_input_image(frame, input_dimension)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if cuda:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img))
            output = write_results(output, confidence, len(classes), overlap_threshold)

            if type(output) == int:
                frames += 1
                print("FPS of the _video is {:5.4f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(input_dimension))

            im_dim = im_dim.repeat(output.size(0), 1) / input_dimension
            output[:, 1:5] *= im_dim

            list(map(lambda x: draw(x, frame, classes[int(x.numpy()[-1])], colors[int(x.numpy()[-1])], True), output))
            key = cv2.waitKey(1)
            cv2.imshow(video_file, frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break
    capture.release()
    cv2.destroyAllWindows()

else:
    """
                                        Image Object Detection
    """
    images = args.load
    batch_size = int(args.bs)
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    # Make the save file directory if not existing
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    loaded_images = [cv2.imread(x) for x in imlist]

    # Process all loaded images into pytorch variables
    image_batches = list(map(prepare_input_image, loaded_images, [input_dimension for x in range(len(imlist))]))

    # Create tensor for dimensions list
    image_dimensions = [(x.shape[1], x.shape[0]) for x in loaded_images]
    image_dimensions = torch.FloatTensor(image_dimensions).repeat(1,2)

    if cuda:
        image_dimensions = image_dimensions.cuda()

    if batch_size != 1:
        # Number of batches is the number of images divided by the batch size and add 1 if remainder
        num_batches = len(imlist) // batch_size + bool(len(image_dimensions) % batch_size)
        # Reformat processed images into sub lists for each batch
        image_batches = [torch.cat((image_batches[i * batch_size : min((i + 1)*batch_size,
                                                    len(image_batches))])) for i in range(num_batches)]
    write = 0
    start_detections = time.time()
    for i, batch in enumerate(image_batches):
        start_batch = time.time()

        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch))

        prediction = write_results(prediction, confidence, len(classes), overlap_threshold)

        end_batch = time.time()

        if type(prediction) == int:
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                     (end_batch - start_batch)/batch_size))
                print("{0:20s} {1:s}\n".format("Objects Detected:", ""))

        try:
            prediction[:, 0] += i * batch_size
        except TypeError:
            continue

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end_batch - start_batch)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))

        if cuda:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print ("No detections were made")
        exit()

    # Rescale bbox corners from nn input dimensions back to image dimensions
    image_dimensions = torch.index_select(image_dimensions, 0, output[:, 0].long())
    scale = torch.min(input_dimension/image_dimensions,1)[0].view(-1,1)
    # Size x and y coordinates to padded image
    output[:,[1,3]] -= (input_dimension - scale * image_dimensions[:,0].view(-1,1)) / 2
    output[:,[2,4]] -= (input_dimension - scale * image_dimensions[:,1].view(-1,1)) / 2
    # Sized to original image
    output[:,1:5] /= scale

    # Clamp bboxs that extend outside the image boundary to the image boundary
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, image_dimensions[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, image_dimensions[i,1])

    list(map(lambda x: draw(x, loaded_images, classes[int(x.numpy()[-1])], colors[int(x.numpy()[-1])]), output))

    det_names = pd.Series(imlist).apply(lambda x: "{}/yolo_{}".format(args.save, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, loaded_images))