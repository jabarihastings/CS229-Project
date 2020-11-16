"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torchvision.models as models

model_mapping = {
    'alexnet': models.alexnet(pretrained=True), # fast, used for debugging
    'resnet': models.resnet18(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    # 'inception': models.inception_v3(pretrained=True),  # Buggy inception
    'mobilenet': models.mobilenet_v2(pretrained=True),
    'googlenet': models.googlenet(pretrained=True),
}

def get_num_outputs(args):
        if args.net == 'vgg16' or args.net == 'alexnet':
            return model_mapping[args.net].classifier[6].out_features
        elif args.net == "mobilenet":
            return model_mapping[args.net].classifier[1].out_features
        else:
            return model_mapping[args.net].fc.out_features

def get_model(args):
    if args.net in model_mapping:
        return model_mapping[args.net]
    return None


class Net(nn.Module):

    def __init__(self, args, params):
        super(Net, self).__init__()
        self.pretrained = get_model(args)
        self.dropout_rate = params.dropout_rate if hasattr(params, 'dropout_rate') else 0.0
        self.my_new_layers = nn.Sequential(nn.Linear(get_num_outputs(args), 128, bias=True),
                                           nn.ReLU(),
                                           nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(128, 3, bias=True),
                                           nn.LogSoftmax(dim=1))
        # why add layers versus doing  model_ft.classifier = nn.Linear(num_ftrs, num_classes) ?


    def forward(self, s):
        s = self.pretrained(s)
        s = self.my_new_layers(s)
        return s
