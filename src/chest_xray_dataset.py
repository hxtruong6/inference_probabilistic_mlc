import deeplake
import torch
import torchvision.transforms as transforms


FIXED_SEED_RANDOM = 6


def extract_features(model, dataloader, device):
    model = model.to(device)
    model.eval()
    extracted_features = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(f"\U0001F4A1 Processing batch: {i+1}/{len(dataloader)}: {data}")
            inputs, labels = data["image"], data["findings"]
            inputs = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            extracted_features.append({"features": outputs, "labels": labels})
    return torch.cat(extracted_features)


def main():
    torch.manual_seed(FIXED_SEED_RANDOM)

    print("\U0001F4A1 Loading dataset...")
    ds = deeplake.load("hub://activeloop/nih-chest-xray-test")
    dataloader = ds.pytorch(
        num_workers=8,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\U0001F4BB Using device: {device}")

    print("\U0001F4A1 Loading model...")
    model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
    # Default output layer is 1000, we need to replace it with other number as per our requirement

    # Replace the fully-connected layer with a new one that has 512 output features
    # model.fc = torch.nn.Linear(
    #     model.fc.in_features, 512
    # )

    print("\U0001F4A1 Extracting features...")
    # what is the output shape of the model?
    features = extract_features(model, dataloader, device)
    print(
        f"\U0001F4A1 Features shape: {features['features'].shape} Labels shape: {features['labels'].shape}"
    )

    # print the first 5 labels
    print(features["labels"][:5])
    print(features["features"][:5])


if __name__ == "__main__":
    main()
