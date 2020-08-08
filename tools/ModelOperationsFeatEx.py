import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

class Model_Operations():
    def __init__(self, model):
        self.model = model
        self.outputs = []
        self.register_hooks()

    def feature_hook(self, module, input, output):
        self.outputs.append(output)

    def register_hooks(self):
        self.model.avgpool.register_forward_hook(self.feature_hook)
        self.model.fc.register_forward_hook(self.feature_hook)


    def __call__(self, x):
        self.outputs = []
        _ = self.model(x)
        features,Logit = self.outputs
        return features,Logit
