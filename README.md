# 🧠 MED-02: Brain Tumor Classification AI

This project is a medical imaging analysis tool designed to classify brain tumors from MRI scans. It leverages a combination of image processing and machine learning to provide predictions with confidence scores and visual saliency maps.

## ✨ Features
- **MRI Image Upload**: Upload MRI scans in PNG, JPG, or JPEG format.
- **Ensemble Prediction**: Uses multiple model perspectives (Intensity, Texture, and Hybrid) to provide a robust final ensemble prediction.
- **Visual Analysis**: Generates a saliency/gradient-based heatmap to highlight the regions of interest in the MRI scan.
- **Medical Metrics**: Provides image-specific metrics like Mean, Std Dev, and Entropy.
- **Comparison Tool**: Compare multiple uploaded scans in a tabular format and export the results to CSV.
- **Performance Dashboard**: View validation metrics and a confusion matrix for the underlying models.

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Image Processing**: [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/)
- **Model Logic**: Custom Python implementation in `brain_tumor_ai` module.

## 🚀 Getting Started

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kishore302408/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run brain_tumor_mri_app.py
   ```

### Deployment on Streamlit Cloud
This app is optimized for Streamlit Cloud. To deploy:
1. Connect your GitHub account to [share.streamlit.io](https://share.streamlit.io/).
2. Select the `Brain-Tumor-Detection` repository.
3. Set the main file path to `brain_tumor_mri_app.py`.
4. Click **Deploy**.

## 📊 Classification Categories
The model classifies MRI scans into four categories:
- ✅ **No Tumor**
- 🔴 **Glioma**
- 🔵 **Meningioma**
- 🟢 **Pituitary**

---
*Disclaimer: This tool is for educational/demonstration purposes only and should not be used for clinical diagnosis.*
