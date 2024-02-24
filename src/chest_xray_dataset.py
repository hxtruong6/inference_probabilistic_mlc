import numpy as np
import pandas as pd
import skimage
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms  # Still needed for basic transforms
import torchvision

SEED = 6


def extract_features(model, dataset, device):
    """Extracts features from a given model on a TorchXRayVision dataset."""
    model.to(device)
    model.eval()

    features = []  # Storage for features and labels
    with torch.no_grad():
        for sample in dataset:
            images = sample["img"].to(device)
            labels = sample["lab"].to(
                device
            )  # Assuming labels are in multi-label format

            outputs = model(images)
            features.append({"features": outputs.cpu(), "labels": labels.cpu()})

    return features  # List of dictionaries with 'features' and 'labels'


def load_dataloader():
    imgpath = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/images-224"
    csvpath = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017.csv"

    # Image loading and preprocessing
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resize to match ResNet18 input
            transforms.CenterCrop(224),  # Crop for ResNet18
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard ImageNet normalization
        ]
    )

    dataset = torchvision.datasets.ImageFolder(
        root=image_folder, transform=data_transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False
    )  # Adjust batch size as needed

    # # Load Dataset using TorchXRayVision
    # dataset = xrv.datasets.NIH_Dataset(
    #     imgpath=imgpath,
    #     # csvpath=csvpath,
    #     transform=transforms.Compose(
    #         [
    #             xrv.datasets.XRayCenterCrop(),  # Preprocessing specific to X-rays
    #             xrv.datasets.XRayResizer(224),
    #         ]
    #     ),
    # )
    # print(dataset.csvpath)
    # print(dataset.pathologies)
    # print(dataset.csv.head())

    # print(f"\U0001F4A1 Dataset size: {len(dataset)}")
    # # # Filtering out non-existent image files from CSV (if any)
    # # # filter_testset(dataset)
    # # print(f"\U0001F4A1 After filtering - Dataset size: {len(dataset)}")

    return dataset


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4BB Using device: {device}")

    dataloader = load_dataloader()

    print(f"\U0001F4A1 Dataloader size: {len(dataloader)}")

    # Load a pre-trained model
    model = xrv.models.DenseNet(
        weights="densenet121-res224-nih",
        # apply_sigmoid=True,
    )  # NIH chest X-ray8
    model.features = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],
        torch.nn.AvgPool2d((7, 7)),
        torch.nn.Flatten(),
    )

    extract_features = []

    for i, data in enumerate(dataloader):
        if i == 2:
            break

        img = data["img"]
        # print(img.shape)
        feat_vec = model.features(img)
        # print(feat_vec.shape)

        # outputs = model(img)
        # print(outputs.shape)
        # print(dict(zip(model.pathologies, outputs[0].detach().numpy())))

        extract_features.append(
            {
                "idx": data["idx"].detach().numpy(),
                "features": feat_vec.detach().numpy(),
                "labels": data["lab"].detach().numpy(),
            }
        )

    # get only 8 label. check index of label in dataset.pathologies by order then drop the rest column index.

    print(extract_features[0])


if __name__ == "__main__":
    main()
