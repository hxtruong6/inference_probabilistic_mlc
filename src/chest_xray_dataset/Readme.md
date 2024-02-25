# NIH Chest X-Ray Feature Extraction Package

## **Overview**

This package provides code and utilities for:

* Preprocessing NIH Chest X-ray images.
* Extracting features from the NIH Chest X-Ray dataset using a pre-trained DenseNet model.
* Optionally, modifying the feature extractor with global pooling or convolutional layers to adjust output dimensionality.
* Saving and loading extracted features to NumPy arrays for further analysis.
* Converting extracted features and labels into Pandas DataFrames.

## **Prerequisites**

* Python 3.x
* PyTorch
* TorchXRayVision
* NumPy
* Pandas
* Pillow (PIL)

### **Installation**

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/nih-chest-xray-analysis
   ```

2. Navigate to the project directory:

   ```bash
   cd nih-chest-xray-analysis 
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt 
   ```

### **Dataset**

* Download the NIH Chest X-Ray dataset and the corresponding labels CSV file from [Link NIH Chest X-ray Dataset (Resized to 224x224)](https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0)
* Place the data in the following structure within your project:

   ```md
   datasets/
     NIH/
       dataset/  (Image files)
       NIH_Data_Entry_2017__testset_image_labels.csv 
   ```

## **Usage**

1. **Modify Base Directory (If Needed):** If your dataset is in a different location, adjust the `BASE_DIR` variable in the code.

2. **Run the Script:**

   ```bash
   python chest_xray_dataset.py 
   ```

   This will preprocess the images, extract features, save features and labels to NumPy arrays, and load them into DataFrames.

### **Explanation of Code**

* **chest_xray_dataset.py:**  Contains the core logic for data loading, preprocessing, model setup, feature extraction, and saving/loading features.

* **chest_xray_utils.py:** Contains helper functions for feature and label data manipulation.

### **Customization**

* **Feature Extractor:** Modify the `set_custom_feature_extractor` function to experiment with different feature extraction strategies (global pooling, convolutional reduction).

* **Downstream Tasks:** Load the extracted features (NumPy arrays or DataFrames) and use them for classification, clustering, or other analyses.

#### **Custom Transform Class**

To adapt image preprocessing to the specific requirements of TorchXRayVision models, I've written a custom transform class. Here's a comparison highlighting the key differences:

**TorchXRayVision Original**
*Original Code*
[https://github.com/hxtruong6/inference_probabilistic_mlc/assets/24609363/630a68d1-50ec-44ea-a048-f4d4ac03e20c](https://github.com/hxtruong6/inference_probabilistic_mlc/assets/24609363/630a68d1-50ec-44ea-a048-f4d4ac03e20c)

**My Custom Transform**
*Custom Code*
[https://github.com/hxtruong6/inference_probabilistic_mlc/assets/24609363/37ef28e3-2612-4854-a939-95ba92c459ff](https://github.com/hxtruong6/inference_probabilistic_mlc/assets/24609363/37ef28e3-2612-4854-a939-95ba92c459ff)

## **Contributing**

We welcome contributions! Feel free to open pull requests or issues for improvements and bug fixes.

## **Contact**

For questions or support, please reach out to [Email1](mailto:hxtruong6@gmail.com) or [Email2](mailto:hxtruong@jaist.ac.jp).
