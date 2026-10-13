## 🧾 Overview 

Project ini berfokus ke mengklasifikasi emosi manusia dari ekspresi wajah menggunakan deep learning dan transfer learning technniques. Menggunakan arsitektur Convulutional Neural Network (CNN) yang mengandung (ConvNeXt, EfficientNet, ResNet, dan VGG) yang secara sistematik mengevaluasi dan membandingkan untuk mengidentifikasi model yang paling efektif untuk task ini. Model ini melakukan traning dan testing di custom-build RGB dataset gambar yang membandingkan 7 kelas emosi: `Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, dan `Surprise`. Performa model ini menggunakan key matrics seperti akurasi, F1-score dan Confusion Matrics, sementara visualisasi Grad-CAM digunakan untuk meningkatkan interpretabillitas dengan meng-highlight daerah yang mempengaruhi prediksi model

## 🗂️ Project Structure 
```bash
emotion-face-recognition/
│
├── Dataset/                      # 📃 Not included in the repository duue to exceeding maximum size
│
├── models/                       # 🤖 Not included in the repository duue to large size 
│                          
├── src/                          # 🧠 Core source code (preprocessing, training, testing & visualization)
│
├── app.py                         # 🌐 Web app (Gradio deployment files)
│
├── requirements.txt               # 📃 Python dependencies
│
└── README.md                      # 📘 Project documentation
```

## 🗃️ Dataset

Untuk dataset, project ini menggunakan dataset [FER_25 dataset](https://www.kaggle.com/datasets/shaikhborhanuddin/fer-25) yang terdiri dari **7,200 Gambar RGB berResolusi tinggi (224x224)** dengan berbagai kategory: Anger, Disgust, Fear, Happiness, Neutral, Sadness, dan Surprise. Setiap kategory mengandung kurang dari 925-945 gambar traning dan 96-125 gambar tes.

## 🖥️ How to Use

- Copy this repo
```bash
git clone https://github.com/justRaam-exe/emotion-face-recognition.git
```

- Download Dataset dan model
[Kaggle](https://www.kaggle.com/datasets/shaikhborhanuddin/fer-25) & 
[google drive](https://drive.google.com/drive/folders/1fawjtIIxCBRv8H-JiBYYg88DraSdq3Sb?usp=sharing)

- Gunakan Virtual Environment (```venv```)
```bash
python3 -m venv venv
source venv/bin/activate
```

- Install ```Requirements.txt```
```bash
pip install requirements.txt
```

- Jalankan ```main.py```
```bash
python3 main.py
```