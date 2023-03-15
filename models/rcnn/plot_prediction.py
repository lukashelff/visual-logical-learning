import os

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes

from michalski_trains.dataset import michalski_categories, rcnn_michalski_categories


def plot_prediction(model, dataloader, device):
    idx = dataloader.dataset.indices[0]
    torch_image, box = dataloader.dataset.dataset.__getitem__(idx)
    img_path = dataloader.dataset.dataset.get_image_path(idx)
    img = dataloader.dataset.dataset.get_pil_image(idx)
    # img = read_image(img_path)
    # pil image to tensor
    img = to_tensor(img) * 255
    # float tensor image to int tensor image
    img = img.to(torch.uint8)

    model.eval()
    torch_image = torch_image.to(device).unsqueeze(0)
    model.to(device)
    prediction = model(torch_image)[0]
    labels = [rcnn_michalski_categories()[i] for i in prediction["labels"]]
    boxes = [i for i in prediction["boxes"]]
    for c in range(len(labels)):
        box = draw_bounding_boxes(img, boxes=prediction["boxes"][c:c + 1],
                                  labels=labels[c:c + 1],
                                  colors="red",
                                  width=2, font_size=15)
        im = to_pil_image(box.detach())
        # save pil image
        pth = f'output/models/rcnn/test_prediction/boxes/im_{idx}_mask_{c}.png'
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        im.save(pth)
    for c in range(len(labels)):
        box = draw_bounding_boxes(img, boxes=prediction["masks"][c:c + 1],
                                  labels=labels[c:c + 1],
                                  colors="red",
                                  width=2, font_size=15)
        im = to_pil_image(box.detach())
        # save pil image
        pth = f'output/models/rcnn/test_prediction/masks/im_{idx}_mask_{c}.png'
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        im.save(pth)

    # im.show()
