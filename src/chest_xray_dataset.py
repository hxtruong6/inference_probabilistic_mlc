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


def main():
    torch.manual_seed(SEED)

    imgpath = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/images-224"
    csvpath = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4BB Using device: {device}")

    # Load Dataset using TorchXRayVision
    # dataset = xrv.datasets.NIH_Dataset(
    #     imgpath=imgpath,  # Update with your NIH dataset path
    #     csvpath=csvpath,
    #     transform=transforms.Compose(
    #         [
    #             xrv.datasets.XRayCenterCrop(),  # Preprocessing specific to X-rays
    #             xrv.datasets.XRayResizer(224),
    #             # transforms.ToTensor(),
    #         ]
    #     ),
    # )

    img = skimage.io.imread(imgpath + "/00000013_008.png")
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...]  # Make single color channel

    transform = torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )

    img = transform(img)
    img = torch.from_numpy(img)

    # Load a pre-trained model
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")  # NIH chest X-ray8

    feat_vec = model.features(img[None, ...])

    print(feat_vec.shape)
    # print(feat_vec[:5, :5, :5])

    # get labels
    outputs = model(img[None, ...])
    print(outputs.shape)
    print(dict(zip(model.pathologies, outputs[0])))

    # print("\U0001F4A1 Extracting features...")
    # all_features = extract_features(model, dataset, device)

    # # Access extracted features and labels
    # for data in all_features:
    #     print(
    #         f"Features shape: {data['features'].shape} Labels shape: {data['labels'].shape}"
    #     )

    #     # Example: Print first 5 labels and features
    #     print(data["labels"][:5])
    #     print(data["features"][:5])


if __name__ == "__main__":
    main()
