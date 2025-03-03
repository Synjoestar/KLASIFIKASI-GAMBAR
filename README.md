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

**Dataset Source:** [Kaggle - Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  

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
```

## ⚙️ Installation
### 1️⃣ Clone the repository
---
git clone https://github.com/Synjoestar/KLASIFIKASI-GAMBAR.git
cd KLASIFIKASI-GAMBAR
---

2️⃣ Install dependencies
bash
Salin
Edit
pip install -r requirements.txt
🏋️ Training the Model
Run the Jupyter Notebook (notebook.ipynb) to train the model. The model will:

✅ Load and preprocess the dataset
✅ Train using a CNN architecture
✅ Implement data augmentation and callbacks
✅ Save the trained model in multiple formats

📈 Model Evaluation
🔹 Accuracy and loss curves are plotted for visualization.
🔹 A classification report and confusion matrix are generated.
🔹 The model is validated using test images.

🚀 Exporting the Model
The trained model is saved in multiple formats for different use cases:

📌 SavedModel (TensorFlow)
python
Salin
Edit
model.save("saved_model_rice")
📌 TensorFlow Lite (TFLite)
python
Salin
Edit
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_rice")
tflite_model = converter.convert()
with open("tflite/model.tflite", "wb") as f:
    f.write(tflite_model)
📌 TensorFlow.js (TFJS)
bash
Salin
Edit
pip install tensorflowjs
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='Predictions' --saved_model_tags=serve saved_model_rice tfjs_model
🔍 Inference
Perform inference using TFLite
python
Salin
Edit
import tensorflow.lite as tflite
from PIL import Image
import numpy as np

interpreter = tflite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

image_path = "sample_image.jpg"  # Replace with actual image
input_size = input_details[0]['shape'][1]  
input_data = preprocess_image(image_path, input_size)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
print(f"Predicted Class: {predicted_class}")
