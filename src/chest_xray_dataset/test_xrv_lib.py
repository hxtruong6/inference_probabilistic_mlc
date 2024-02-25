import torchxrayvision as xrv
import skimage, torch, torchvision

# Prepare the image:
img = skimage.io.imread(
    "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/dataset/images-224/00000003_000.png"
)
print(f"Original image : {img}")
# print(img.shape)
img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range

print(img.shape)
# what is.mean(2) ?  mean(2) is the mean of the 3rd dimension of the image
# img = img.mean(2)[None, ...]  # Make single color channel
img = img[None, :, :]
print(f"Image shape: {img.shape}")

transform = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
)

img = transform(img)
img = torch.from_numpy(img)

print(f"Transformed image shape: {img.shape}")
print(f"Transformed image: {img}")

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
outputs = model(img[None, ...])  # or model.features(img[None,...])

# Print results
print(dict(zip(model.pathologies, outputs[0].detach().numpy())))

feat_vec = model.features(img[None, ...])
print(f"Feature vector shape: {feat_vec.shape}")
print(f"Feature vector: {feat_vec}")
