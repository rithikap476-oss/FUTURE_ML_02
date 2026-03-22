# 🎫 FUTURE_ML_02 — Support Ticket Classification & Prioritization
### Machine Learning Task 2 (2026) | Future Interns

Automatically classify support tickets into **categories** and assign **priority levels** using NLP + ML.

---

## 🚀 Run in VS Code (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# 3. View plots in the /plots/ folder
```

That's it. **Everything runs from a single file.**

---

## 📁 Project Structure

```
FUTURE_ML_02/
├── main.py            ← Single runnable script (entire pipeline)
├── requirements.txt   ← Dependencies
├── README.md          ← This file
├── plots/             ← Auto-created — 6 visualizations saved here
└── models/            ← Auto-created — 5 trained model files saved here
```

---

## 🧠 ML Pipeline (inside main.py)

| Step | What Happens |
|------|-------------|
| **1. Data** | Generates 2,000 realistic support tickets |
| **2. EDA** | 6-panel dashboard: distributions, heatmaps, trends |
| **3. Preprocess** | Lowercase → strip HTML/URLs → stopwords → lemmatise |
| **4. TF-IDF** | 5,000 features, unigrams + bigrams |
| **5. Train** | Logistic Regression, Naive Bayes, Linear SVC, Random Forest |
| **6. Evaluate** | Accuracy, Precision, Recall, F1 (macro) |
| **7. Plots** | Confusion matrix, feature importance, cross-validation |
| **8. Predict** | Live predictions on 10 sample tickets |
| **9. Save** | All models persisted as `.pkl` files |

---

## 📊 Output Plots

| File | Contents |
|------|---------|
| `plots/01_eda_dashboard.png` | EDA: distributions, word counts, priority heatmap |
| `plots/02_model_comparison.png` | Accuracy / Precision / Recall / F1 for all 4 models |
| `plots/03_confusion_category.png` | Confusion matrix for category classification |
| `plots/03_confusion_priority.png` | Confusion matrix for priority classification |
| `plots/04_features_category.png` | Top discriminative TF-IDF features per category |
| `plots/04_features_priority.png` | Top discriminative TF-IDF features per priority |
| `plots/05_cross_validation.png` | 5-fold CV scores |
| `plots/06_prediction_summary.png` | Live prediction category & priority breakdown |

---

## 🏷️ Categories & Priorities

**Categories** → Billing · Technical Issue · Account · General Query

**Priorities** → 🔴 High · 🟡 Medium · 🟢 Low

---

## 💼 Using a Real Dataset (Kaggle)

Replace the `generate_dataset()` call in `main.py` Step 1 with:

```python
df = pd.read_csv("your_file.csv")
df = df.rename(columns={
    "your_text_col"    : "ticket_text",
    "your_category_col": "category",
    "your_priority_col": "priority",
})
```

Recommended datasets:
- https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
- https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset
