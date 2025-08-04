# 🌾 Paddy Disease Classification

This repository contains the code for a **Paddy Disease Classification** application built with **TensorFlow**, **Keras**, and **Streamlit**. The app uses **EfficientNet**, **fine-tuning** and **transfer learning** to classify images of rice leaves into one of ten disease categories.

---

## 📋 Project Structure

```
├── app.py                    # Streamlit application code
├── requirements.txt          # Python dependencies
├── rice_disease_model.keras  # Trained Keras model
├── balanced_images/          # Folders of labeled images (10 classes)
│   ├── bacterial_leaf_blight/
│   ├── bacterial_leaf_streak/
│   ├── bacterial_panicle_blight/
│   ├── blast/
│   ├── brown_spot/
│   ├── dead_heart/
│   ├── downy_mildew/
│   ├── hispa/
│   ├── normal/
│   └── tungro/
└── Paddy_Disease_Classification.ipynb  # Exploratory and training notebook
```

---

## 🚀 Features

* Upload a rice leaf image via a web interface
* Classify into one of 10 categories:

  * bacterial\_leaf\_blight
  * bacterial\_leaf\_streak
  * bacterial\_panicle\_blight
  * blast
  * brown\_spot
  * dead\_heart
  * downy\_mildew
  * hispa
  * normal
  * tungro
* Display prediction label and confidence score
* Easy-to-use Streamlit UI

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MuhammadHaweras/Paddy_Disease_Classification.git
   cd Paddy_Disease_Classification
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App Locally

```bash
streamlit run app.py
```

* Open the URL printed in the console (usually `http://localhost:8501`).
* Upload a rice leaf image to see the predicted disease and confidence.

---

## 📊 Training & Evaluation

* See **`Paddy_Disease_Classification.ipynb`** for:

  * Data preprocessing
  * Model training with EfficientNet and transfer learning
  * Fine-tuning strategies and evaluation metrics
  * Confusion matrix and classification report

---

## 💾 Model

* **`rice_disease_model.keras`** is the saved Keras model (EfficientNet-based)
* To load in Python:

  ```python
  import tensorflow as tf
  model = tf.keras.models.load_model("rice_disease_model.keras")
  ```

---


* Dataset and inspiration from PlantVillage and other rice disease research.
* Built and maintained by **Muhammad Haweras**. Feel free to connect on [LinkedIn](https://www.linkedin.com/in/muhammad-haweras-7aa6b11b2/) or check out my work on [GitHub](https://github.com/MuhammadHaweras).
