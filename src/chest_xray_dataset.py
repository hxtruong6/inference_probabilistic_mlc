import os
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms  # Still needed for basic transforms
import torchvision
from torch.utils.data import DataLoader

SEED = 6

import torchvision.transforms as transforms
import numpy as np
from PIL import Image


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
class MultiLabelImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, df_labels, is_valid_file=None):
        super().__init__(root, transform, is_valid_file=is_valid_file)
        self.df_labels = df_labels
        print(f"\U0001F4A5 Loading dataset from: {root}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # sample = XRCVImageTransform()(sample)

        filename = os.path.basename(path)
        # check if filename is in valid_filenames

        labels = (
            self.df_labels[self.df_labels["Image Index"] == filename]
            .iloc[0]
            .tolist()[1:]
        )  # Get the list of labels
        return sample, torch.tensor(labels)  # Return image and one-hot tensor


def is_valid_file(path, valid_filenames):
    file_name = os.path.basename(path)
    return file_name in valid_filenames


def load_dataloader():
    image_folder = (
        "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/dataset"
    )
    label_file = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH_Data_Entry_2017__testset_image_labels.csv"

    df_labels = pd.read_csv(label_file)
    valid_filenames = df_labels["Image Index"].values

    # Image loading and preprocessing
    data_transform = transforms.Compose([XRCVImageTransform()])

    dataset = MultiLabelImageFolder(
        root=image_folder,
        transform=data_transform,
        df_labels=df_labels,
        is_valid_file=lambda file_name: is_valid_file(file_name, valid_filenames),
    )
    print(f"\U0001F4A1 Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4BB Using device: {device}")

    dataloader = load_dataloader()
    print(f"\U0001F4C1 Dataloader size: {len(dataloader)}")

    # Load a pre-trained model
    model = xrv.models.DenseNet(
        weights="densenet121-res224-nih",
        # apply_sigmoid=True,
    )  # NIH chest X-ray8
    # model.features = torch.nn.Sequential(
    #     *list(model.classifier.children())[:-1],
    #     torch.nn.AvgPool2d((7, 7)),
    #     torch.nn.Flatten(),
    # )

    # print(f"feature size: {model.features}")

    extract_features = []

    with torch.no_grad():
        for data in dataloader:
            print(f"\U0001F4E6 Processing batch... {data[0].shape}")
            image, labels = data
            print(f"Image 1 transformed by custom: {image[0]}")

            image = image.to(device)

            # The first dimension is the batch size
            feat_vec = model.features(image)

            print(f"\U0001F4E7 Feature vector shape: {feat_vec.shape}")

            print(f"Input batch image shape: {image.shape}")

            outputs = model(image)
            print(f"Output batch shape: {outputs.shape}")
            print(dict(zip(model.pathologies, outputs[0].detach().numpy())))

            # append feat_vec to extract_features in each instance of batch
            for i in range(labels.shape[0]):
                extract_features.append(
                    {
                        "features": feat_vec[i].cpu(),
                        "labels": labels[i].cpu(),
                    }
                )
            break

    # get only 8 label. check index of label in dataset.pathologies by order then drop the rest column index.
    # print(extract_features)


if __name__ == "__main__":
    main()
