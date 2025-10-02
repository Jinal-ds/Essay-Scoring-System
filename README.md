# Essay-Scoring-System
Automated Essay Scoring (AES) system powered by the BERT-Base model. Leverages sophisticated NLP preprocessing pipelines (tokenization, cleaning, feature engineering) to achieve high-accuracy, context-aware scoring of essays.

This project implements an **Automated Essay Scoring (AES) system** using **BERT** for feature extraction and a **Streamlit web app** for user interaction. The system is trained on the [Learning Agency Lab - Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) dataset from Kaggle.

---

## 📂 Project Structure

```
Essay-Scoring-System/
│
├── BERT_MODEL.ipynb    # Jupyter Notebook for model training & evaluation
├── app.py              # Streamlit web app for essay scoring
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

---

## 📊 Dataset

Due to size limitations, the dataset is **not included** in this repository.
You can download it from the Kaggle competition:
👉 [Learning Agency Lab - AES 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)

Place the dataset in the working directory before running the notebook.

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/Essay-scoring-system.git
cd Essay-scoring-system
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model

Open **`BERT_MODEL.ipynb`** and run all cells.

* This notebook loads the dataset, trains the BERT-based essay scoring model, and saves it for inference.

### 2. Run the Web App

Start the Streamlit app:

```bash
streamlit run app.py
```

This will launch the AES system in your browser.
You can paste an essay into the text box, and the system will generate a predicted score.

---

## 🛠️ Requirements

Main dependencies (full list in `requirements.txt`):

* Python 3.8+
* PyTorch / TensorFlow
* Transformers (HuggingFace)
* Streamlit
* scikit-learn
* pandas, numpy

---

## 📌 Notes

* The dataset is not included due to size constraints. Please download it from Kaggle.
* The system currently uses a fine-tuned BERT model; performance may vary depending on training.

---

## 📜 License

This project is for **educational purposes only**.
