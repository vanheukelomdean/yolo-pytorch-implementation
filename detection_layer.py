import torch.nn as nn

class DetectionLayer(nn.Module):
    """
        Final layer for detection overriden to store detection box anchors
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors