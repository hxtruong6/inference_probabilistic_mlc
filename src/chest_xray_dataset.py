import deeplake
import torch
import torchvision.transforms as transforms

FIXED_SEED = 6  # For consistency


def extract_features(model, dataloader, device):
    """Extracts features from a given model on a dataloader."""
    model.to(device)
    model.eval()
    all_features = []  # More descriptive name

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):  # Unpacking for readability
            print(f"\U0001F4A1 Processing batch: {i+1}/{len(dataloader)}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_features.append({"features": outputs, "labels": labels})

    return torch.cat(all_features)


def main():
    torch.manual_seed(FIXED_SEED)

    print("\U0001F4A1 Loading dataset...")
    ds = deeplake.load("hub://activeloop/nih-chest-xray-test")
    dataloader = ds.pytorch(
        num_workers=8,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                # Add resizing if needed: transforms.Resize(224)
                transforms.Resize(224),
            ]
        ),
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Specify CUDA device
    print(f"\U0001F4BB Using device: {device}")

    print("\U0001F4A1 Loading model...")
    model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)

    # Replace the fully-connected layer
    # model.fc = torch.nn.Linear(model.fc.in_features, 512)

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
