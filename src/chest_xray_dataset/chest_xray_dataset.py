import os
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

from chest_xray_utils import (
    load_df_features_from_npy,
    load_features_from_npy,
    save_features_to_npy,
)

SEED = 6
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"\U0001F4C1 Base project folder: {BASE_DIR}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\U0001F4BB Using device: {DEVICE}")


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
        print(f"\U0001F4A5 Loading dataset from: {root}")

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
    print(f"\U0001F4A1 Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def load_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-nih",
        # apply_sigmoid=True,
    )  # NIH chest X-ray8

    return model


def set_custom_feature_extractor(model: xrv.models.DenseNet):
    model.features = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],
        # Option 1 - Use the following layers
        torch.nn.AdaptiveAvgPool2d((512, 1)),
        torch.nn.Flatten(),
        # Option 2 - Use Conv2d - recheck the output size
        # torch.nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
        # torch.nn.Flatten(),
    )

    return model


def get_features(model, dataloader):
    extract_features = []
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(f"\U0001F4E6 Processing batch {i}... | Batch size: {data[0].shape}")
            image, labels = data
            # print(f"Image 1 transformed by custom: {image[0]}")

            image = image.to(DEVICE)
            # The first dimension is the batch size
            feat_vec = model.features(image)
            # print(f"\U0001F4D1 Feature vector shape: {feat_vec.shape}")

            # append feat_vec to extract_features in each instance of batch
            for i in range(labels.shape[0]):
                extract_features.append(
                    {
                        "features": feat_vec[i].cpu(),
                        "labels": labels[i].cpu(),
                    }
                )

            # TODO: for testing purpose, break after 1 batch
            # break

    return extract_features


def main():
    dataloader = load_dataloader(batch_size=32, shuffle=False)
    print(f"\U0001F4C1 Dataloader size: {len(dataloader)}")

    model = load_model()
    model = set_custom_feature_extractor(model)

    # get only 8 label. check index of label in dataset.pathologies by order then drop the rest column index.
    extract_features = get_features(model, dataloader)
    print(
        f"\U0001F4E7 Extracted features of The first image: \n {extract_features[0]}\n"
    )

    save_features_to_npy(
        extract_features,
        filename=f"{BASE_DIR}/datasets/nih_feature_vectors.npy",
    )

    print(
        f"\U0001F4E8 Saved features to file: \n{BASE_DIR}/datasets/nih_feature_vectors.npy"
    )
    data_feat = load_features_from_npy(
        features_filename=f"{BASE_DIR}/datasets/nih_feature_vectors.npy",
        labels_filename=f"{BASE_DIR}/datasets/nih_feature_vectors__labels.npy",
    )

    print(f"\U0001F4E9 Loaded features: \U0001F4C0 {data_feat[0].shape}")

    df_feats, df_labels = load_df_features_from_npy(
        features_filename=f"{BASE_DIR}/datasets/nih_feature_vectors.npy",
        labels_filename=f"{BASE_DIR}/datasets/nih_feature_vectors__labels.npy",
    )

    print(f"\U0001F4B9 Loaded features: \n{df_feats.head(5)}")
    print(f"\U0001F3B9 Loaded labels: \n{df_labels.head(5)}")


if __name__ == "__main__":
    main()
