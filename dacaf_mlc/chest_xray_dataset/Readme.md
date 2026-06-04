# NIH ChestX-ray14 feature extraction

This subpackage produces the `nih_feature_vectors_{densenet,resnet,resnetae}.npy`
files consumed by `dacaf_mlc.evaluate` for the
`chest_xray_nih__{densenet,resnet,resnetae}` rows of the paper.

The pipeline turns raw NIH ChestX-ray14 PNGs into pooled feature vectors
extracted from three pretrained backbones (DenseNet121, ResNet50, ResNetAE).
Eight of NIH's 14 pathology labels are kept; each backbone is reduced to a
512-dim feature vector via the custom pooling head defined in
`XRVModel._set_custom_feature_extractor`.

## Prerequisites

```
pip install torch torchvision torchxrayvision pandas numpy pillow tqdm scikit-image
```

GPU strongly recommended (CPU extraction over the full 112k-image test set
is impractical).

## One-time data preparation

1. **Download the NIH ChestX-ray14 dataset** (resized 224×224 mirror is
   sufficient and faster):
   <https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0>

2. **Lay out** the images and metadata under the project root:

   ```
   datasets/
     NIH/
       dataset/                       # PNGs (or a single ImageFolder-compatible subdir)
       Data_Entry_2017.csv            # raw NIH metadata
       test_list.txt                  # NIH-provided official test split
   ```

3. **Build the one-hot label CSV** that the extractor reads,
   `datasets/NIH_Data_Entry_2017__testset_image_labels.csv` (shipped in this
   repo). To regenerate it from the raw NIH metadata: strip the trailing "Y" in
   `Patient Age`, subset to the official `test_list.txt` split, and one-hot the
   `Finding Labels` to the 8 kept pathologies, writing the column convention
   `Image Index, label_1, …, label_8`.

## Extract features (run once per backbone)

```bash
python dacaf_mlc/chest_xray_dataset/chest_xray_dataset.py --model densenet
python dacaf_mlc/chest_xray_dataset/chest_xray_dataset.py --model resnet
python dacaf_mlc/chest_xray_dataset/chest_xray_dataset.py --model resnetae
```

Each invocation writes two files into `datasets/`:

- `nih_feature_vectors_<model>.npy`            — shape `(N, 512)` features
- `nih_feature_vectors_<model>__labels.npy`    — shape `(N, 8)`   one-hot labels

These are the inputs the paper's eval pipeline picks up via
`load_df_features_from_npy` in `read_datasets_from_folder`.

## Files

| File | Purpose |
|---|---|
| `chest_xray_dataset.py` | Main extraction script. CLI: `--model {densenet,resnet,resnetae}`. |
| `chest_xray_utils.py` | `load_df_features_from_npy`, `save_features_to_npy`, etc. Imported by the eval pipeline. |
