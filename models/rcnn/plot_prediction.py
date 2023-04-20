import os

import torch
from torchvision.io import read_image
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from michalski_trains.dataset import rcnn_blender_categories


def predict_and_plot(model, dataloader, device):
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
    labels = [rcnn_blender_categories()[i] for i in prediction["labels"]]
    boxes = [i for i in prediction["boxes"]]
    for c in range(len(labels)):
        box = draw_bounding_boxes(img, boxes=prediction["boxes"][c:c + 1],
                                  labels=labels[c:c + 1],
                                  colors="red",
                                  width=2, font_size=15)
        im = to_pil_image(box.detach())
        # save pil image
        pth = f'output/models/rcnn/test_prediction/boxes/im_{idx}_bbox_{c}.png'
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        im.save(pth)
    for c in range(len(labels)):
        mask = prediction["masks"][c]
        mask[mask > .5] = 1
        mask[mask <= .5] = 0
        mask = mask.to(torch.bool)
        box = draw_segmentation_masks(img, masks=mask,
                                      colors="red",
                                      )
        im = to_pil_image(box.detach())
        # save pil image
        pth = f'output/models/rcnn/test_prediction/masks/im_{idx}_mask{c}_{labels[c]}.png'
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        im.save(pth)


def plot_mask(prediction, identifier, pil_img, tag='', output_dir='output/models/multi_label_rcnn/masks_prediction'):
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    # )
    # tensor_image = inv_normalize(tensor_image)
    # tensor to image
    # img = tensor_image * 255


    # transform = transforms.Compose([transforms.ToTensor()])
    # img = transform(pil_img).to('cpu')
    img = to_tensor(pil_img) * 255
    # float tensor image to int tensor image
    img = img.to(torch.uint8).to('cpu')
    labels = [rcnn_blender_categories()[i] for i in prediction["labels"]]
    boxes = [i for i in prediction["boxes"]]
    # for c in range(len(labels)):
    #     box = draw_bounding_boxes(img, boxes=prediction["boxes"][c:c + 1],
    #                               labels=labels[c:c + 1],
    #                               colors="red",
    #                               width=2, font_size=15)
    #     im = to_pil_image(box.detach())
    #     # save pil image
    #     pth = f'output/models/rcnn/plt_pred/boxes/im_{identifier}_bbox_{c}.png'
    #     os.makedirs(os.path.dirname(pth), exist_ok=True)
    #     im.save(pth)
    for c in range(len(labels)):
        mask = prediction["masks"][c]
        mask[mask > .5] = 1
        mask[mask <= .5] = 0
        mask = mask.to(torch.bool).to('cpu')
        box = draw_segmentation_masks(img, masks=mask)
        im = to_pil_image(box.detach())
        # save pil image
        tag = tag if tag == '' else '/' + tag
        pth = f'{output_dir}/masks{tag}/im_{identifier}/{labels[c]}_mask{c}.png'
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        im.save(pth)
