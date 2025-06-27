# AI-Generated Text Detection Using the RAID Dataset

This project focuses on detecting AI-generated text using machine learning and transformer-based models. It leverages the RAID (Real or AI-generated Discourse) dataset, containing millions of human-written and synthetic text samples, to build reliable classifiers.

## 📊 Project Overview

The notebook implements a pipeline to:

- **Randomly Sample & Balance Data:** Select approximately 75,000 entries from over 5 million records, ensuring a balanced dataset for fair training (human-written vs. AI-generated).
- **Preprocessing:** Clean and prepare the text data for modeling.
- **Modeling Approaches:**
  - Baseline Model: TF-IDF with Logistic Regression.
  - Advanced Model: Fine-tuned RoBERTa Transformer for improved detection performance.
- **Evaluation:** Compare models using standard classification metrics.

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression)
- HuggingFace Transformers (RoBERTa)
- Google Colab (for large-scale processing)

## 📁 Dataset

The project uses the **RAID (Real or AI-generated Discourse) dataset**, which contains a mix of human-written and AI-generated text. Due to size constraints, a sampled and balanced subset (~75,000 records) is used for demonstration and modeling purposes.

> **Note:** Full dataset access may require appropriate permissions or downloads from the original source.

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have Python 3.7+ and necessary libraries installed.
3. Place the dataset (`train.csv`) in the correct directory or update the paths in the notebook.
4. Run the notebook step by step, either locally or using Google Colab.

## ⚡ Results

The notebook provides performance comparisons between traditional ML techniques and transformer-based models, highlighting the effectiveness of deep learning in distinguishing AI-generated text.

## 📬 Author

**Ranjitha Narasimhamurthy**  
University of California Riverside 
Course: Data Mining Techniques (235)

---

