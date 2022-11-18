from detectron2.data import (
    MetadataCatalog, build_detection_train_loader,
)
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from models.detectron import register_ds
from util import *
from pycocotools import mask as maskUtils


def plot_masked_im(im, label):
    image = torch2numpy(im[:3, :, :])
    mask = im[3, :, :]
    plt.imshow(image, 'gray', interpolation='none')
    plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.title(f'michalski train image with label {label}')
    plt.show()


# show an image from the michalski train dataset
def show_im(train_ds):
    ds = train_ds.dataset
    # im, label = ds.__getitem__(0)
    # im = torch2numpy(im)
    # fig, ax = plt.subplots()
    # ax.imshow(im)
    # train = ds.get_m_train(0)
    # plt.title(f'michalski train image with label {label.numpy()}')
    # ax.set_axis_off()
    # plt.show()
    # os.makedirs('output/test_images/', exist_ok=True)
    # fig.savefig('output/test_images/train1.png', bbox_inches='tight', pad_inches=0)

    for id in range(10):
        im = ds.get_pil_image(id)
        label = ds.get_label_for_id(id)
        plt.imshow(im)
        plt.title(f'michalski train image with label {label}')
        plt.show()
        plt.close()


# show an image from the michalski train dataset including mask of first car
def show_masked_im(train_ds):

    os.makedirs('output/test_images/', exist_ok=True)
    for im_id in range(train_ds.__len__()):
    # for im_id in range(1):
        im = train_ds.get_pil_image(im_id)
        masks = train_ds.get_mask(im_id)
        rle = masks['car_2']['mask']
        mask = maskUtils.decode(rle)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(im, 'gray', interpolation='none')
        ax.imshow(mask, 'jet', interpolation='none', alpha=0.7)
        # plt.title(f'michalski train image with overlaid mask')
        fig.savefig(f'output/test_images/{im_id}masked_train.png', bbox_inches='tight', pad_inches=0, dpi=387.2)
        plt.close()


        fig, ax = plt.subplots()
        bbox = maskUtils.toBbox(rle)
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
                         facecolor='none')
        ax.add_patch(rect)
        ax.set_axis_off()
        ax.imshow(im, 'gray', interpolation='none')
        # plt.title(f'michalski train image with overlaid mask')
        fig.savefig(f'output/test_images/{im_id}boxed_train.png', bbox_inches='tight', pad_inches=0, dpi=387.2)
        plt.close()

    # plt.imshow(im)
    # plt.show()
    # plt.close()
    #
    # plt.title('michalski train w/ bounding box of the first car')
    # train = train_ds.get_m_train(0)
    # bbox = maskUtils.toBbox(rle)
    # rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
    #                  facecolor='none')
    # ax.add_patch(rect)
    # ax.set_axis_off()
    #
    # plt.imshow(m)
    # plt.show()
    # fig.savefig('output/test_images/train1.png', bbox_inches='tight', pad_inches=0, dpi=400)


# plot gt instances from dataset frame
def plot_detectron_gt(cfg, base_scene, train_col):
    image_count = 10
    register_ds(base_scene, train_col, image_count)
    metadata = MetadataCatalog.get("michalski_val_ds")
    data_loader = build_detection_train_loader(cfg)
    for data in data_loader:
        instances = data[0]['instances'].to("cpu")
        im_path = data[0]['file_name']
        # im = cv2.imread(im_path)
        im = torch2numpy(data[0]['image'])[:, :, ::-1]
        gt_boxes = instances.gt_boxes
        gt_classes = instances.gt_classes
        gt_masks = instances.gt_masks
        for label, mask in zip(gt_classes, gt_masks):
            title = metadata.thing_classes[int(label)]
            mask = mask.to("cpu").detach().numpy()
            # plot_bitmap(mask, title)
            visualizer = Visualizer(im, metadata=metadata)
            # draw = visualizer.draw_instance_predictions(instance)
            draw = visualizer.overlay_instances(masks=[mask])
            seg_im = draw.get_image()
            path = f'./output/detectron/ground_truth.png'
            plt.imshow(seg_im)
            plt.title(title)
            plt.show()


def plot_rle(rle, title):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title)
    ax.set_axis_off()
    m = maskUtils.decode(rle)
    plt.imshow(m)
    plt.show()


def plot_bitmap(bitmap, title):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title)
    ax.set_axis_off()
    plt.imshow(bitmap)
    plt.show()
