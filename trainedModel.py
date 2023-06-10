import os
from tokenize import String
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mrcnn2.mrcnn import utils
from mrcnn2.mrcnn.config import Config
import mrcnn2.mrcnn.model as modellib
from mrcnn2.mrcnn import visualize
from skimage.filters import threshold_mean

# Directory to save logs and trained model
MODEL_DIR = "logs/"

# # Local path to trained weights file
# COCO_MODEL_PATH = "/home/ryzenrtx/Computer Vision/deep-dental-image/ObjectDetection/mask_rcnn_coco.h5"
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


# class ShapesConfig(Config):
#     """Configuration for training on the toy shapes dataset.
#     Derives from the base Config class and overrides values specific
#     to the toy shapes dataset.
#     """
#     # Give the configuration a recognizable name
#     NAME = "shapes"

#     # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
#     # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

#     # Number of classes (including background)
#     NUM_CLASSES = 1 + 1  # background + 3 shapes

#     # Use small images for faster training. Set the limits of the small side
#     # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM = 1024
#     IMAGE_MAX_DIM = 1024

#     # Use smaller anchors because our image and objects are small
#     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

#     # Reduce training ROIs per image because the images are small and have
#     # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
#     TRAIN_ROIS_PER_IMAGE = 32

#     # Use a small epoch since the data is simple
#     STEPS_PER_EPOCH = 100

#     # use small validation steps since the epoch is small
#     VALIDATION_STEPS = 30


class TeethConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "teeth"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 10

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # BG has 80 classes


config = TeethConfig()
# config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig,ax


class TeethDataset(utils.Dataset):

    def __init__(self, img_dataset_path, mask_dataset_path):

        self.img_dataset_path = img_dataset_path
        self.mask_dataset_path = mask_dataset_path
        super(TeethDataset, self).__init__()

    def load_teeth(self):
        # Add classes
        self.add_class("teeth", 1, "tooth")
        # print(img_dataset)
        count = 0
        id_imgs = []
        for root, dirs, files in os.walk(self.img_dataset_path):
            id_imgs = [i for i in range(
                0, len(os.listdir(self.img_dataset_path)), 1)]
            # print(id_imgs)
            id_paths = []
            for i in id_imgs:
                id_paths.append(os.path.join(root, "img"+str(i)+".jpg"))
            # print(id_paths)
            masks_paths = []
            for i in id_imgs:
                masks_paths.append(os.path.join(
                    self.mask_dataset_path, "mask"+str(i)+".png"))
            # print(masks_paths)
            for image_id, image_path, mask_path in zip(id_imgs, id_paths, masks_paths):
                if os.path.exists(mask_path):
                    self.add_image("teeth",
                                   image_id=count,
                                   path=image_path,
                                   image_path_id=image_id)
                    count += 1
        print("{} teeth loaded.".format(count))
        # return image_id

    def load_mask(self, image_id):
        # info = self.image_info[image_id]
        path_mask = self.mask_dataset_path + "mask" + str(image_id) + ".png"
        image_mask = cv2.imread(path_mask)
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        #image = gray2rgb(image)
        ret, thresh = cv2.threshold(image_mask, 0, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        class_ids = np.ones(len(contours), np.int32)
        masks = np.zeros(
            (image_mask.shape[0], image_mask.shape[1], len(contours)), dtype=np.bool)
        for i in range(len(contours)):
            img_cont = np.zeros(
                (image_mask.shape[0], image_mask.shape[1], 3), dtype=image_mask.dtype)
            cv2.drawContours(img_cont, contours, i, (255, 255, 255), -1)
            bimask = np.zeros(img_cont.shape[0:2], dtype=np.uint8)
            bimask = img_cont[:, :, 0]
            thresh = threshold_mean(bimask)
            mask = bimask > thresh
            masks[:, :, i] = mask
        # print(masks.shape)
        return masks, class_ids

# to load model and saved weights


def load_saved_model(config=TeethConfig(), log_path="logs/", model_path="logs/mask_rcnn_teeth1.h5", mode="inference"):
    # load MRCNN configuration
    model = modellib.MaskRCNN(mode=mode,config=config,model_dir=log_path)
    # load already saved weights on custom dataset
    model.load_weights(model_path,by_name=True)
    return model

# to make prediction
def make_prediction(model,image_path=str):
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # print(original_image.shape)
    try:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    image = cv2.imread(image_path)
    if image.shape[2] != 3:
        try:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        except Exception as e:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        except Exception as e: 
            print(e)
    results = model.detect([image], verbose=1)
    return results,original_image

# to visualize the prediction
def make_visualization(original_image,results,detection_classes=['BG','tooth'],figAx=None):
    fig, ax = visualize.display_instances(original_image, results[0]['rois'], 
                                          results[0]['masks'], results[0]['class_ids'],
                                          detection_classes, results[0]['scores'], figAx=get_ax(),show_caption=False)
    fig.canvas.draw()

    img_with_detection = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

    img_with_detection = img_with_detection.reshape(
                                fig.canvas.get_width_height()[::-1] + (3,))
    return img_with_detection
