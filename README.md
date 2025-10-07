README - AI Trace Finder Project

 Overview
      AI Trace Finder is a machine learning pipeline and application developed for forensic scanner identification and tamper detection from scanned document images. The project leverages residual image analysis and hybrid deep learning + classical ML models to determine image acquisition scanners and detect image tampering.

Step 1: Mount and Set Up the Dataset  
- Mounted Google Drive for persistent storage using “google.colab.drive”.  
- Cleanly unmounted and remounted to prevent stale mounts.  
- Dataset organized into multiple folders:  
- Flatfield images (scanner residual calibration data)  
- Official and Wikipedia scanned images (real-world samples)  
- Originals and Tampered images for train/test  

 Step 2: Exploratory Data Analysis (EDA)  
- Counted total files per dataset folder.  
- Examined image formats, sizes, aspect ratios, and distributions.  
- Visualized sample images per scanner and category.  
- Analyzed pixel intensity distributions across classes.  

 Step 3: Data Preprocessing and Residual Extraction  
- Applied wavelet denoising on grayscale images to extract residual noise patterns (PRNU).  
- Generated residuals by subtracting denoised images from originals.  
- Saved residual images per scanner for later fingerprint extraction.
- Converted PDF originals to TIFF image format (300 DPI) for consistency.  
- Used pixel-level LBP and FFT features for texture/noise characterization.

 Step 4: Fingerprint Extraction  
- Computed average noise residual per scanner class (fingerprints).  
- Saved fingerprints and deterministic ordered key lists for reproducible results.  
- Visualized noise fingerprints with enhanced contrast and color maps.

 Step 5: Feature Extraction  
- Extracted features per image residual combining:  
- Correlations between image residual and known fingerprints (PRNU correlation).  
- Radial FFT frequency-band energy features.  
- Local Binary Pattern histograms capturing texture.  
- Advanced enhanced features optionally include texture gradient and statistical metrics.

 Step 6: Dataset Formatting, Splitting & Augmentation  
- Created structured datasets containing image residuals, numeric features, and labels.  
- Applied stratified-group k-fold cross-validation to ensure unbiased train/val splits.  
- Performed patch extraction and data augmentations on residuals for diversity.  
- Balanced clean and tampered image domains to reduce bias in tamper detection training.

 Step 7: Model Training  
- Trained a hybrid CNN model combining:  
- Residual high-pass filtered image input processed via convolutional layers.  
- Complementary handcrafted PRNU+FFT+LBP feature vector input via fully connected layers.  
- Used categorical cross-entropy loss and Adam optimizer with learning rate schedules.  
- Trained for multiple epochs with early stopping and best model checkpointing.  
- Persisted trained model, label encoder, and feature scaler artifacts.

 Step 8: Tamper Patch Detection Model  
- Using extracted patches from residual images, calculated patch-level features as in Step 5.  
- Trained a calibrated Support Vector Machine with RBF kernel on patch features.  
- Performed threshold optimization using ROC curves and Youden's J statistic on validation data.  
- Saved scaler, SVM model, and calibrated decision thresholds as tamper detection artifacts.

 Step 9: Paired Tamper Detection  
- Created patch-difference features from aligned paired images to detect subtle copy-move, splicing, and retouching manipulations.  
- Calibrated paired SVM classifiers and determined per-type decision thresholds.  
- Saved paired models and thresholds separately.

 Step 10: Model Evaluation and Visualization  
- Evaluated scanner hybrid model final accuracy, loss, and ROC AUC metrics on test data.  
- Visualized training history curves, t-SNE and PCA feature clusters, and scanner noise maps with matplotlib/seaborn.  
- Generated confusion matrices and detailed per-class classification reports.  
- Exported inference results into CSV for audit and analysis.

 Step 11: Inference and Deployment  
- Built a Streamlit app with file upload support for image/PDF inputs.  
- Performed residual preprocessing and feature extraction live on uploads.  
- Applied hybrid CNN scanner prediction and tamper patch SVM inference.  
- Displayed annotated scanner predictions and tamper verdicts with confidence scores.  
- Provided user-friendly dashboard with emojis and styled HTML panels.

 Libraries and Tools Used  
- Google Colab for interactive development and Drive storage mount.  
- OpenCV, PyWavelets, scikit-image for image preprocessing and feature extraction.  
- TensorFlow Keras for hybrid CNN training.  
- scikit-learn for classical ML SVM models and calibration.  
- Streamlit for web dashboard application.  
- Matplotlib and Seaborn for EDA visualizations.


 
Data and File Structure (in Github)  

AI-Trace-Finder/
├── .gitignore                                  #Ignore .venv, __pycache__, etc.
│   
├── App/                                       # Core application code
│   ├── app.py                              # Streamlit UI entry point
│   ├── inference.py                     # Scanner + tamper detection logic
│   ├── utils/                              
│   │   └── preprocess.py                   # Feature extraction or preprocessing
│   └── models/                                  # All model & artifact files
│       ├── scanner_hybrid.keras
│       ├── hybrid_label_encoder.pkl
│       ├── hybrid_feat_scaler.pkl
│       ├── scannerfingerprints.pkl
│       ├── fp_keys.npy
│       ├── artifacts_tamper_patch/
│       │   ├── patch_scaler.pkl
│       │   ├── patch_svm_sig_calibrated.pkl
│       │   └── thresholds_patch.json
│       └── artifacts_tamper_pair/
│           ├── pair_scaler.pkl
│           ├── pair_svm_sig.pkl
│           └── pair_thresholds_topk.json
│
├── data/
│   └── manifests/
│       └── tamper_manifest_grouped.csv     #Manifest used for tamper reference
│
├── AI_Trace_Finder.ipynb                 # Training / experimentation notebook
├── LICENSE                                      # License
├── README.md                               # Project overview and instructions
├── requirements.txt                            # Final dependency list
├── runtime.txt                                    # Python version spec for deployment



Deployment(in Streamlit)

App(deployment) link: https://aiscanneridentifier.streamlit.app/


 


  Summary  
           This project demonstrates an end-to-end pipeline for analyzing scanned document images, extracting noise-based scanner fingerprints, training a hybrid CNN and classical ML models (SVM) for scanner identification and tamper detection. The approach integrates wavelet-based residual extraction, handcrafted texture and frequency domain features, linked with deep learning and statistical model calibration for forensic accuracy.

