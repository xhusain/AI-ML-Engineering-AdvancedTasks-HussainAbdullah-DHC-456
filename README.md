# AI-ML-Engineering-AdvancedTasks-HussainAbdullah-DHC-456

This repository contains three advanced machine learning tasks, each demonstrating different techniques in natural language processing and predictive modeling.

---

## **Task 1: News Topic Classification using BERT**

### **Objective**

Fine-tune a transformer model (e.g., BERT) to classify news headlines into topic categories.

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

### **Required Libraries**

```bash
pip install torch transformers datasets scikit-learn
```

---

## **Task 2: Telco Customer Churn Prediction**

### **Objective**

Build a reusable and production-ready machine learning pipeline for predicting customer churn.

### **Dataset**

* [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
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

### **Required Libraries**

```bash
pip install pandas scikit-learn joblib
```

---

## **Task 5: Support Ticket Auto-Tagging using LLMs**

### **Objective**

Automatically tag support tickets into categories using a large language model (LLM).

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

### **Required Libraries**

```bash
pip install pandas datasets transformers scikit-learn openai gradio torch
```

**Launch Gradio App**:

```python
iface.launch()
```
