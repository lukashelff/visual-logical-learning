import timm
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models.cnns.multi_label_nn import MultiLabelNeuralNetwork
from models.mlp.mlp import MLP
from models.multioutput_regression.pos_net import PositionNetwork
from models.set_transformer import SetTransformer
from models.spacial_attr_net.attr_net import AttributeNetwork


def get_model(model_name, pretrained, num_output, num_class):
    model_image_processing = None
    if model_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif model_name == 'resnet101':
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
    elif model_name == 'VisionTransformer':
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=2)
    elif model_name == 'EfficientNet':
        # model = models.efficientnet_b7(pretrained=pretrained, num_classes=2)
        # model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=2)
        # model = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained, num_classes=2)
        model = timm.create_model('tf_efficientnetv2_l_in21k', pretrained=pretrained, num_classes=2)
    elif model_name == 'set_transformer':
        model = SetTransformer(dim_input=32, dim_output=num_output * num_class)
    elif model_name == 'rcnn':
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=22, autoshape=False)
        # weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        # model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

        # model initialization
        weights = models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        # model_image_processing = models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()

        model = models.detection.maskrcnn_resnet50_fpn_v2(weights=weights,
                                                          image_mean=[0.485, 0.456, 0.406],
                                                          image_std=[0.229, 0.224, 0.225],
                                                          # num_classes=22 + 20,
                                                          rpn_batch_size_per_image=256,
                                                          # box_score_thresh=0.9
                                                          )
        # for predicting masks
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # define a new head for the detector with required number of classes, 22 for the label specific classes and 20 as the upper bound for the number of cars which can be present in a scene
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 22 + 4)
        # for predicting boxes
        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 22 + 4)

    elif model_name == 'attr_predictor':
        model = AttributeNetwork(dim_input=32)
    elif model_name == 'pos_predictor':
        model = PositionNetwork(dim_input=4, dim_output=num_output)
    elif model_name == 'MLP':
        model = MLP(dim_in=4 * 32, dim_out=num_output * num_class)
    else:
        raise AssertionError('select valid model')

    if 'resnet' in model_name:
        if num_class == 2:
            num_ftrs = model.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            model.fc = nn.Linear(num_ftrs, num_class)
        else:
            model = MultiLabelNeuralNetwork(model, num_output)
    return model, model_image_processing