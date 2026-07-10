# 🎬 IMDb Movie Review Sentiment Analysis using Simple RNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-SimpleRNN-FF6F00?style=for-the-badge\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge\&logo=streamlit)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=for-the-badge\&logo=googlecolab)
![IMDb Dataset](https://img.shields.io/badge/Dataset-IMDb-01B4E4?style=for-the-badge)

### Deep Learning based sentiment analysis system built using Embedding Layers and Simple RNN architecture.

🔗 **Live Demo:** https://imdb-movie-review-sentiment-analysis-6e59fvmedyu49n6oeprsid.streamlit.app/


</div>

---

# 📌 Project Overview

Understanding customer opinions from text data is one of the most important Natural Language Processing (NLP) applications.

This project uses a **Simple Recurrent Neural Network (Simple RNN)** with an **Embedding Layer** to classify IMDb movie reviews as either:

* 😊 Positive Review
* 😠 Negative Review

The model learns contextual relationships between words and predicts sentiment based on the overall meaning of the review.

The complete project pipeline, including model training, experimentation, evaluation, Streamlit testing, and GitHub integration, was developed using **Google Colab with GPU acceleration**.

---

# 🚀 Live Demo

### 🌐 Streamlit Application

https://imdb-movie-review-sentiment-analysis-6e59fvmedyu49n6oeprsid.streamlit.app/

---

# ✨ Features

* 🎬 Movie review sentiment classification
* 🧠 Simple RNN architecture
* 🔤 Learnable Embedding Layer
* 📖 IMDb dataset support
* 📏 Sequence padding for fixed-length input
* ⚡ Real-time sentiment prediction
* 🌐 Interactive Streamlit interface
* ☁️ Cloud-based development workflow
* 🚀 Google Colab GPU training support
* 📦 End-to-end preprocessing pipeline

---

# 🧠 Model Architecture

```text
Movie Review
      ↓
Tokenization
      ↓
Integer Encoding
      ↓
Sequence Padding
      ↓
Embedding Layer
      ↓
Simple RNN Layer
      ↓
Dense Layer
      ↓
Sigmoid Activation
      ↓
Positive / Negative Prediction
```

---

# 📚 Dataset Information

Dataset Used:

**IMDb Movie Review Dataset**

Dataset Statistics:

* 50,000 movie reviews
* 25,000 training samples
* 25,000 testing samples
* Binary classification problem
* Balanced dataset

---

# 🔤 Text Preprocessing Pipeline

The following preprocessing steps are performed:

### 1️⃣ Lowercase Conversion

```text
This Movie Is Amazing
↓
this movie is amazing
```

### 2️⃣ Word to Integer Mapping

```text
movie → 14
amazing → 530
great → 84
```

### 3️⃣ Sequence Padding

All reviews are padded to a fixed length of:

```text
500 tokens
```

to make them compatible with RNN input requirements.

---

# 🏗️ Model Architecture Details

| Layer               | Configuration            |
| ------------------- | ------------------------ |
| Embedding Layer     | Vocabulary Size = 10,000 |
| Embedding Dimension | 128                      |
| Simple RNN Layer    | 128 Units                |
| Output Layer        | 1 Neuron (Sigmoid)       |
| Loss Function       | Binary Crossentropy      |
| Optimizer           | Adam                     |

---

# ☁️ Google Colab Workflow

Since local GPU resources were unavailable, the complete project lifecycle was executed using **Google Colab GPU environments**.

This included:

* Dataset experimentation
* Model training
* Hyperparameter tuning
* Streamlit application testing
* Local tunnel testing
* GitHub repository management

The Streamlit application was tested directly inside Google Colab using tunneling services before deployment.

---

# 📂 Project Structure

```text
IMDb-Sentiment-Analysis/
│
├── app.py
├── imdb_simple_rnn.h5
├── experiments.ipynb
├── prediction.ipynb
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

# 🛠️ Tech Stack

| Category                | Technology                |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning           | TensorFlow / Keras        |
| NLP                     | Tokenization & Embeddings |
| Model Architecture      | Simple RNN                |
| Frontend                | Streamlit                 |
| Development Environment | Google Colab              |
| Version Control         | Git & GitHub              |

---

# 📈 Model Output

Example Prediction:

```text
Review:
"This movie was absolutely fantastic and had brilliant acting."

Prediction:
😊 Positive Review

Confidence:
96.84%
```

---

# 📸 Application Preview

## Home Screen

<img width="455" height="412" alt="home screen" src="https://github.com/user-attachments/assets/88f1f618-bf9b-40af-9c02-2849c234bc3b" />

## Positive Prediction Example

<img width="440" height="281" alt="positive prediction" src="https://github.com/user-attachments/assets/c133689a-7b5e-4017-b18d-bc34a20a5576" />


## Negative Prediction Example

<img width="450" height="277" alt="negative prediction" src="https://github.com/user-attachments/assets/833e5986-3a8f-4bd4-af81-b60b321bf89e" />


---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/chaudhary-dhruv/IMDb-movie-review-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

* LSTM Implementation
* GRU Implementation
* Bidirectional LSTM
* Attention Mechanism
* Transformer Architecture
* Explainable AI using SHAP
* Multi-class sentiment classification

---

# 👨‍💻 Author

## Dhruv

Machine Learning | Deep Learning | NLP | Generative AI

GitHub:
https://github.com/chaudhary-dhruv/

LinkedIn:
https://www.linkedin.com/in/dhruv-chaudhary-8026a3258/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps improve visibility and supports future development.

---
