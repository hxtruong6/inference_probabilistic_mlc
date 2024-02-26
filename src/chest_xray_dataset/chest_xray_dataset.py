import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from chest_xray_utils import (
    load_df_features_from_npy,
    load_features_from_npy,
    save_features_to_npy,
)

SEED = 6
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(f"\U0001F4C1 Base project folder: {BASE_DIR}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"\U0001F4BB Using device: {DEVICE}")


class XRCVImageTransform(object):

    def __call__(self, img):
        """Applies the series of transformations to the input image for X-ray images from NIH dataset.
        This is following the torchxrayvision library's default preprocessing steps.

        Args:
            img: Input image (PIL Image or NumPy array)

        Returns:
            Transformed image (PyTorch Tensor)
        """

        # 1. Convert to NumPy array (if necessary)
        if isinstance(img, Image.Image):
            img = np.array(img)

        # 2. Convert to [-1024, 1024] range
        img = (img / 255 * 2048) - 1024

        # 3. Make single color channel (assuming grayscale is intended)
        if len(img.shape) > 2:
            img = img.mean(axis=2)

        # 4. Center Crop
        img = self.crop_center(img)

        # 5. Resize to 224
        img = Image.fromarray(img).resize((224, 224))

        # 6. Convert to PyTorch Tensor
        img = transforms.ToTensor()(img)

        return img

    def crop_center(self, img: np.ndarray) -> np.ndarray:
        y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[starty : starty + crop_size, startx : startx + crop_size]


# Create dataset (Modified)
class NIHChestXRayDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, df_labels, is_valid_file=None):
        super().__init__(root, transform, is_valid_file=is_valid_file)
        self.df_labels = df_labels
        logger.info(f"\U0001F4A5 Loading dataset from: {root}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        filename = os.path.basename(path)
        labels = (
            self.df_labels[self.df_labels["Image Index"] == filename]
            .iloc[0]
            .tolist()[1:]
        )  # Get the list of labels
        return sample, torch.tensor(labels)  # Return image and one-hot tensor


class XRVModel:
    MODEL_TYPE = ["densenet", "resnet", "ResNetAE"]

    def __init__(self, model, dataloader, model_type):
        self.model = model
        self.model_type = model_type
        self._set_custom_feature_extractor()

        self.dataloader = dataloader
        self.extracted_features = []

    def _set_custom_feature_extractor(self):
        if self.model is None:
            raise ValueError("Model is not loaded")

        if isinstance(self.model, xrv.models.DenseNet):
            # [1024, 7, 7] -> [1, 512]
            self.custom_features = torch.nn.Sequential(
                # torch.nn.AdaptiveAvgPool1d((512, 1)),
                torch.nn.AvgPool2d(kernel_size=7, stride=7),
                torch.nn.Flatten(),
            )
            logger.info(f"\U0001F4D1 Model custom_features: {self.custom_features}")
        elif isinstance(self.model, xrv.models.ResNet):
            # [2048, 1] -> [1, 512]
            self.custom_features = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.Flatten(),
            )

            logger.info(f"\U0001F4E6 Model custom_features: {self.custom_features}")
        elif self.model_type == "ResNetAE":
            # Convert model.encode to a feature extractor [512, 3, 3] to -> [1, 512]
            assert isinstance(self.model, xrv.autoencoders._ResNetAE)
            self.custom_features = torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=3, stride=3),
                torch.nn.Flatten(),
                # torch.nn.Linear(512 * 3 * 3, 512),
            )

            logger.info(f"\U0001F4B2 Model custom_features: {self.custom_features}")

        else:
            self.custom_features = None

    def _custom_forward(self, x):
        if self.model_type not in self.MODEL_TYPE:
            raise ValueError("Model type is not supported for _custom_forward()")

        if self.custom_features is not None:
            return self.custom_features(x)
        return x

    def extract_features_vec(self):
        self.model.to(DEVICE)
        self.model.eval()
        if self.custom_features is not None:
            self.custom_features.to(DEVICE)

        self.extracted_features: List[Dict[str, Any]] = (
            self._extract_features_from_dataloader()
        )

    def _extract_features_from_dataloader(self) -> List[Dict[str, Any]]:
        extracted_features: List[Dict[str, Any]] = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                logger.info(
                    f"\U0001F4E6 Processing batch {i}... | Batch size: {data[0].shape}"
                )
                image, labels = data
                image = image.to(DEVICE)
                feat_vec = self.model.features(image)
                logger.info(f"\U0001F3D1 Feature vector shape: {feat_vec.shape}")
                feat_vec = self._custom_forward(feat_vec)
                logger.info(
                    f"\U0001F4D1 Feature vector shape: {feat_vec.shape} after custom"
                )

                for i in range(labels.shape[0]):
                    extracted_features.append(
                        {
                            "features": feat_vec[i].cpu(),
                            "labels": labels[i].cpu(),
                        }
                    )

        return extracted_features

    def get_features_vec(self):
        return self.extracted_features

    def save(self, filepath):
        save_features_to_npy(
            extract_features=self.extracted_features,
            filename=filepath,
        )


def is_valid_file(path, valid_filenames):
    file_name = os.path.basename(path)
    return file_name in valid_filenames


def load_dataloader(batch_size=1, shuffle=False):
    image_folder = f"{BASE_DIR}/datasets/NIH/dataset"
    label_file = f"{BASE_DIR}/datasets/NIH_Data_Entry_2017__testset_image_labels.csv"

    df_labels = pd.read_csv(label_file)
    valid_filenames = df_labels["Image Index"].values

    # Image loading and preprocessing
    data_transform = transforms.Compose([XRCVImageTransform()])

    dataset = NIHChestXRayDataset(
        root=image_folder,
        transform=data_transform,
        df_labels=df_labels,
        is_valid_file=lambda file_name: is_valid_file(file_name, valid_filenames),
    )
    logger.info(f"\U0001F4A1 Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def main():
    dataloader = load_dataloader(batch_size=128, shuffle=False)
    logger.info(f"\U0001F4C1 Dataloader size: {len(dataloader)}")

    MODELS = {
        "densenet": {
            "model": xrv.models.DenseNet(weights="densenet121-res224-nih"),
            "model_type": "densenet",
        },
        "resnet": {
            "model": xrv.models.ResNet(weights="resnet50-res512-all"),
            "model_type": "resnet",
        },
        "resnetae": {
            "model": xrv.autoencoders.ResNetAE(weights="101-elastic"),
            "model_type": "ResNetAE",
        },
    }

    # TODO: Select the model
    SELECTED_MODEL = "densenet"
    # SELECTED_MODEL = "resnet"
    # SELECTED_MODEL = "resnetae"

    xrvModel = XRVModel(
        model=MODELS[SELECTED_MODEL]["model"],
        dataloader=dataloader,
        model_type=MODELS[SELECTED_MODEL]["model_type"],
    )
    xrvModel.extract_features_vec()

    # get only 8 label. check index of label in dataset.pathologies by order then drop the rest column index.
    extract_features = xrvModel.get_features_vec()
    logger.info(
        f"\U0001F4E7 Extracted features of The first image: \n {extract_features[0]}\n"
    )

    saved_file = f"{BASE_DIR}/datasets/nih_feature_vectors_{SELECTED_MODEL}.npy"

    xrvModel.save(saved_file)

    logger.info(
        f"\U0001F4E8 Saved features to file: {saved_file} | Total features: {len(extract_features)}\n"
    )

    # ---------------------------   Load the saved features   ---------------------------
    # Load the saved features
    data_feat = load_features_from_npy(saved_file)
    logger.info(f"\U0001F4E9 Loaded features: \U0001F4C0 {data_feat[0].shape}")

    df_feats, df_labels = load_df_features_from_npy(features_filename=saved_file)

    logger.info(f"\U0001F4B9 Loaded features: \n{df_feats.head(5)}")
    logger.info(f"\U0001F3B9 Loaded labels: \n{df_labels.head(5)}")


if __name__ == "__main__":
    main()
