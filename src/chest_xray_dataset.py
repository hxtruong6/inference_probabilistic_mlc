import deeplake
import torch
import torchvision.transforms as transforms
import os

os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of cores in your system

FIXED_SEED = 6  # For consistency
os.environ["ACTIVELOOP_TOKEN"] = (
    "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6Imh4dHJ1b25nNiIsImFwaV9rZXkiOiJmejlETWpUNjZUb3RNNk5GbUJ3TUhhR29Vc255YkJqME1haUVKZXlJY1F1XzQifQ."
)


def extract_features(model, dataloader, device):
    """Extracts features from a given model on a dataloader."""
    model.to(device)
    model.eval()
    all_features = []  # More descriptive name

    with torch.no_grad():
        for i, data in enumerate(dataloader):  # Unpacking for readability
            print(f"\U0001F4A1 Processing batch: {i+1}/{len(dataloader)}")
            print(f"\U0001F4A1 {i+1} - Data: {data}")
            images, labels = data["images"], data["findings"]

            print(f"\U0001F4A1 Batch shape: {images.shape} {labels.shape}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_features.append({"features": outputs, "labels": labels})

    return torch.cat(all_features)


def load_dataset():
    print("\U0001F4A1 Loading dataset...")
    ds = deeplake.load("hub://activeloop/nih-chest-xray-test")
    # ds = deeplake.load("hub://activeloop/nih-chest-xray-train")
    # ds = deeplake.load("hub://activeloop/coco-test")

    # ds = ds.query(
    #     "select * sample by max_weight(contains(findings, 'Hernia'): 20, contains(findings, 'Pneumonia'): 8, contains(findings, 'Fibrosis'): 5, contains(findings, 'Edema'): 5, contains(findings, 'Emphysema'): 2, True: 1) replace True limit 10"
    # )
    # ds = deeplake.load("hub://activeloop/mnist-test")

    # dataloader = (
    #     ds.dataloader()
    #     .transform(
    #         {
    #             "images": transforms.Compose(
    #                 [
    #                     # transforms.Resize(224),
    #                     transforms.ToTensor(),
    #                 ]
    #             ),
    #             "findings": transforms.Compose([transforms.ToTensor()]),
    #         }
    #     )
    #     .batch(4)
    #     .shuffle(False)
    #     .pytorch(
    #         num_workers=4,
    #         # batch_size=4,
    #         # shuffle=False,
    #         # transform=transforms.Compose(
    #         #     [
    #         #         transforms.Resize(224),
    #         #         transforms.ToTensor(),
    #         #     ]
    #         # ),
    #         decode_method={"images": "pil", "findings": "numpy"},
    #     )
    # )
    dataloader = ds.pytorch(
        num_workers=0,
        batch_size=4,
        shuffle=False,
        decode_method={"images": "pil", "findings": "numpy"},
    )

    print("\U0001F4A1 Dataset loaded successfully")
    print(f"Data info: {ds.info}")
    print(f"\U0001F4A1 Number of samples: {len(ds)}")

    # Get a single sample
    sample = ds[0]
    print(f"Sample: \t{sample} ")
    # print(f"Sample image: \n{sample['images'][0]} ")
    # print(f"Sample findings: \n{sample['findings'][0]} ")

    for i, data in enumerate(dataloader):
        print(f"\U0001F4A3 {i+1} - Data: {data}")
        # print(
        #     f"\U0001F4A2 {i+1} - Data: {data['images'].shape} {data['findings'].shape}"
        # )
        if i > 5:
            break
    return dataloader


def load_model():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Specify CUDA device
    print(f"\U0001F4BB Using device: {device}")

    print("\U0001F4A1 Loading model...")
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)

    # Replace the fully-connected layer
    # model.fc = torch.nn.Linear(model.fc.in_features, 512)
    return model, device


def main():
    torch.manual_seed(FIXED_SEED)

    dataloader = load_dataset()
    return

    model, device = load_model()

    print("\U0001F4A1 Extracting features...")
    features = extract_features(model, dataloader, device)

    # ... (rest of your code)
    print(
        f"\U0001F4A1 Features shape: {features['features'].shape} Labels shape: {features['labels'].shape}"
    )

    # print the first 5 labels
    print(features["labels"][:5])
    print(features["features"][:5])


if __name__ == "__main__":
    main()
