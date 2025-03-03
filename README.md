# 🌾 Rice Image Classification  

A deep learning-based image classification model that classifies five types of rice grains using **Convolutional Neural Networks (CNN)**. The model is trained and evaluated using **TensorFlow** and **Keras**, with support for inference using **TensorFlow SavedModel**, **TensorFlow Lite (TFLite)**, and **TensorFlow.js (TFJS)**.  

---

## 📑 Table of Contents  

- [📊 Dataset](#-dataset)  
- [📁 Project Structure](#-project-structure)  
- [⚙️ Installation](#-installation)  
- [🏋️ Training the Model](#-training-the-model)  
- [📈 Model Evaluation](#-model-evaluation)  
- [🚀 Exporting the Model](#-exporting-the-model)  
- [🔍 Inference](#-inference)  
- [📂 Repository](#-repository)  
- [👨‍💻 Contributors](#-contributors)  
- [📜 License](#-license)  

---

## 📊 Dataset  

The dataset used for this project is the **Rice Image Dataset**, which contains five types of rice grains:  

- 🌾 **Arborio**  
- 🌾 **Basmati**  
- 🌾 **Ipsala**  
- 🌾 **Jasmine**  
- 🌾 **Karacadag**  

**Dataset Source:** [Kaggle - Rice Image Dataset](https://www.kaggle.com/datasets)  

---

## 📁 Project Structure  

```bash
submission
├───tfjs_model
│   ├───group1-shard1of1.bin
│   └───model.json
├───tflite
│   ├───model.tflite
│   └───label.txt
├───saved_model
│   ├───saved_model.pb
│   └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
