import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def explain_model_decision(device):
    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=label, **kwargs)
        return tensor_attributions

    print('applying saliency methods to resnet')
    from models.trainer import Trainer
    from PIL import Image as PImage
    from captum.attr import LayerGradCam
    from captum.attr import visualization as viz

    raw_trains = 'MichalskiTrains'
    base_scene = 'base_scene'
    y_val = 'direction'

    num_epochs = 25
    resume = True
    image_count = 8000
    pretrained = True
    model_name = 'resnet18'
    optimizer_ = 'ADAM'
    gamma = .8
    lr = 0.001
    step_size = 10
    trainer = Trainer(base_scene, raw_trains, device, model_name,
                      image_count=image_count, num_epochs=num_epochs, resume=resume, y_val=y_val,
                      pretrained=pretrained, lr=lr, step_size=step_size, gamma=gamma, optimizer_=optimizer_,
                      setup_ds=False, setup_model=False
                      )

    for train_size in [100,1000,8000]:
        out_path = f'output/saliency/train_size_{train_size}/'
        os.makedirs(out_path, exist_ok=True)
        path = trainer.get_model_path(prefix='cv/', suffix='it_0/', im_count=train_size)
        trainer.setup_model(resume=True, path=path)
        model = trainer.model
        for i in range(30):
            full_ds = trainer.full_ds
            image, gt_label = full_ds.__getitem__(i)
            org_image = full_ds.get_pil_image(i)
            image = image.to(device)
            model = model.to(device)
            image.requires_grad = True
            input = image.unsqueeze(0)
            # input.requires_grad = True
            model.eval()
            c, h, w = image.shape
            heat_map = np.random.rand(h, w)
            image_mod = image[None]
            output = model(image_mod)
            _, pred = torch.max(output, 1)
            label = pred.item()

            last_layer = model.layer4
            w, h = int(1920 / 4), int(1080 / 4)
            gco = LayerGradCam(model, last_layer)
            attr_gco = attribute_image_features(gco, input)
            att = attr_gco.squeeze(0).squeeze(0).cpu().detach().numpy()

            gradcam = PImage.fromarray(att).resize((w, h), PImage.ANTIALIAS)
            heat_map = np.asarray(gradcam)

            explained = np.expand_dims(heat_map, axis=2)
            viz.visualize_image_attr(explained, org_image,
                                     sign="positive", method="blended_heat_map",
                                     show_colorbar=False,
                                     cmap='viridis', alpha_overlay=0.6)

            plt.savefig(out_path + f'{i}_saliency_map_gt_{gt_label}_pred_{label}.png', bbox_inches='tight', dpi=400)
            plt.close()
