# AI-ML-Engineering-AdvancedTasks-HussainAbdullah-DHC-456

This repository contains three advanced machine learning tasks, each demonstrating different techniques in natural language processing and predictive modeling.

---

## **Task 1: News Topic Classification using BERT**

### **Objective**

Build a news classifier that predicts the topic of a news article using the **AG News Dataset**. Implement a transformer-based model (BERT) for sequence classification.

### **Dataset**

* [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
* **Classes**: World, Sports, Business, Sci/Tech

### **Approach**

1. **Load Dataset** using Hugging Face Datasets.
2. **Preprocessing**:

   * Rename columns, tokenize text with `BertTokenizerFast`.
   * Pad and truncate sequences to max length 128.
3. **Model**: `bert-base-uncased` fine-tuned for 4-class classification.
4. **Training**:

   * Use Hugging Face `Trainer` API with `TrainingArguments`.
   * Metrics: Accuracy and Weighted F1-score.
5. **Output**:

   * Fine-tuned model saved as `./agnews-bert-classifier`.

### **Results**

* Achieved high accuracy and F1-score on the test set.

### **Run**

```bash
pip install torch transformers datasets scikit-learn
python task1_news_classification.py
```

---

## **Task 2: Telco Customer Churn Prediction**

### **Objective**

Predict customer churn using a structured dataset with both categorical and numerical features. Compare Logistic Regression and Random Forest classifiers.

### **Dataset**

* Telco Customer Churn dataset (CSV).
* Target: **Churn** (Yes = 1, No = 0).

### **Approach**

1. **Load and Clean Data**:

   * Drop `customerID`, encode `Churn`, handle missing values.
2. **Split** dataset into train/test (80/20).
3. **Feature Engineering**:

   * Scale numerical features, one-hot encode categorical features.
4. **Model Selection**:

   * Use `Pipeline` and `GridSearchCV` to tune hyperparameters.
   * Compare Logistic Regression vs Random Forest.
5. **Evaluation**:

   * Accuracy, classification report, and confusion matrix.
6. **Save Best Model**:

   * Saved as `telco_churn_pipeline.pkl`.

### **Run**

```bash
pip install pandas scikit-learn joblib
python task2_telco_churn.py
```

---

## **Task 5: Support Ticket Auto-Tagging using LLMs**

### **Objective**

Automatically assign tags to IT support tickets using **zero-shot**, **few-shot**, and **fine-tuned transformer** methods. Compare approaches and deploy via a Gradio app.

### **Dataset**

* [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
* CSV: `all_tickets_processed_improved_v3.csv`
* Tags represent ticket categories.

### **Approach**

1. **Load Dataset**:

   * Combine `Subject` and `Body` fields, map labels to IDs.
   * Convert to Hugging Face `Dataset`.
2. **Zero-Shot Classification**:

   * Use OpenAI GPT-4 with prompt engineering to predict top 3 tags.
3. **Few-Shot Classification**:

   * Provide a few labeled examples in the prompt for improved accuracy.
4. **Fine-Tuned Transformer**:

   * Train `distilbert-base-uncased` for sequence classification.
   * Evaluate using accuracy and weighted F1-score.
5. **Deployment**:

   * Build Gradio web interface to compare zero-shot, few-shot, and fine-tuned predictions.

### **Run**

```bash
pip install pandas datasets transformers scikit-learn openai gradio torch
python task5_support_ticket_autotagger.py
```

**Launch Gradio App**:

```python
iface.launch()
```

---



