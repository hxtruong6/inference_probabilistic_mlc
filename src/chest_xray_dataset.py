import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms  # Still needed for basic transforms

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


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4BB Using device: {device}")

    # Load Dataset using TorchXRayVision
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="datasets/NIH/NIH/images-224",  # Update with your NIH dataset path
        csvpath="datasets/NIH/Data_Entry_2017__testset.csv",
        transform=transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),  # Preprocessing specific to X-rays
                xrv.datasets.XRayResizer(224),
                # transforms.ToTensor(),
            ]
        ),
    )

    # Load a pre-trained model
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")  # NIH chest X-ray8

    print("\U0001F4A1 Extracting features...")
    all_features = extract_features(model, dataset, device)

    # Access extracted features and labels
    for data in all_features:
        print(
            f"Features shape: {data['features'].shape} Labels shape: {data['labels'].shape}"
        )

        # Example: Print first 5 labels and features
        print(data["labels"][:5])
        print(data["features"][:5])


if __name__ == "__main__":
    main()
