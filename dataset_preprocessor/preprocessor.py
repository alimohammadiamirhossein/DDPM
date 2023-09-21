from .utils import *
from .OxfordPetDataset import *
from torch.autograd import Variable

import torchvision
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import glob
from tqdm import tqdm

from torchvision.datasets import OxfordIIITPet
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms
import torchvision.transforms as tt

# Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from PIL import Image


class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataset = self.dataset_creator()

    def dataset_creator(self):
        merge_trainval_test(self.filepath)
        dataset = pd.read_csv(f"{self.filepath}/annotations/data.txt", sep=" ")
        image_ids = []
        labels = []
        with open(f"{self.filepath}/annotations/trainval.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                labels.append(int(label)-1)

        classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]

        idx_to_class = dict(zip(range(len(classes)), classes))
        dataset['nID'] = dataset['ID'] - 1 #(Note: the original data is between 1-37, not 0-36, we must set these values to the range 0-36 to use the dictionary)
        decode_map = idx_to_class
        dataset["class"] = dataset["nID"].apply(lambda x: decode_label(x, decode_map))

        return dataset

    def preprocessor(self):
        y = self.dataset['class']
        x = self.dataset['Image']

        trainval, x_test, y_trainval, y_test = train_test_split(x, y,
                                                                stratify=y,
                                                                test_size=0.2,
                                                                random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(trainval, y_trainval,
                                                          stratify=y_trainval,
                                                          test_size=0.3,
                                                          random_state=42)

        df_train = pd.DataFrame(y_train)
        ###comment dataset
        df_val = pd.DataFrame(y_val)
        ###comment dataset
        df_test = pd.DataFrame(y_test)
        ###comment dataset

        root_directory = os.path.join(self.filepath)
        images_directory = os.path.join(root_directory, "images")
        masks_directory = os.path.join(root_directory, "annotations", "trimaps")

        train_images_filenames = x_train.reset_index(drop=True)
        val_images_filenames = x_val.reset_index(drop=True)
        test_images_filenames = x_test.reset_index(drop=True)

        # print(" train size: ", len(train_images_filenames),"\n",
        #       "val size: ", len(val_images_filenames),"\n",
        #       "test size: ", len(test_images_filenames))

        ### to show 10 images
        ### different modes

        train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        target_transform = transforms.Compose([transforms.PILToTensor(),
                                               transforms.Resize((256, 256)),
                                               transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor)) ])

        train_dataset = OxfordPetDataset(train_images_filenames,
                                         images_directory,
                                         masks_directory,
                                         transform=train_transform,
                                         transform_mask=target_transform)

        val_dataset = OxfordPetDataset(val_images_filenames,
                                       images_directory,
                                       masks_directory,
                                       transform=train_transform,
                                       transform_mask=target_transform)

        print(torch.cuda.is_available())

        test_transform = A.Compose(
            [A.Resize(256, 256), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        )
        test_dataset = OxfordPetInferenceDataset(test_images_filenames, images_directory, transform=test_transform,)

        output = {
            'images_directory': images_directory,
            'masks_directory': masks_directory,
            'train_images_filenames': train_images_filenames,
            'val_images_filenames': val_images_filenames,
            'test_images_filenames': test_images_filenames,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }

        return output




        ### comment dataset
        # print("About training data:")
        # print(df_train.describe())
        # print("\n****** Class Distribution ******")
        # print(df_train['class'].value_counts())

        ### to show 10 images
        # display_image_grid(train_images_filenames[:10], images_directory, masks_directory)

        ### different modes
        # example_image_filename = train_images_filenames[10]
        # image = plt.imread(os.path.join(images_directory, f'{example_image_filename}.jpg'))
        #
        # resized_image = A.resize(image, height=256, width=256)
        # padded_image = A.pad(image, min_height=512, min_width=512)
        # padded_constant_image = A.pad(image, min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT)
        # cropped_image = A.center_crop(image, crop_height=256, crop_width=256)
        # # InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
        # figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 10))
        # ax.ravel()[0].imshow(image)
        # ax.ravel()[0].set_title("Original image")
        # ax.ravel()[1].imshow(resized_image)
        # ax.ravel()[1].set_title("Resized image")
        # ax.ravel()[2].imshow(cropped_image)
        # ax.ravel()[2].set_title("Cropped image")
        # ax.ravel()[3].imshow(padded_image)
        # ax.ravel()[3].set_title("Image padded with reflection")
        # ax.ravel()[4].imshow(padded_constant_image)
        # ax.ravel()[4].set_title("Image padded with constant padding")
        # plt.tight_layout()
        # plt.show()