"""
╔══════════════════════════════════════════════════════════════════╗
║   SUPPORT TICKET CLASSIFICATION & PRIORITIZATION                ║
║   Machine Learning Task 2 (2026) — Future Interns               ║
║   Run in VS Code: python main.py                                 ║
╚══════════════════════════════════════════════════════════════════╝

Pipeline:
  1. Generate / Load dataset
  2. EDA (Exploratory Data Analysis) + Visualizations
  3. Text Cleaning & Preprocessing
  4. TF-IDF Feature Extraction
  5. Model Training (LR, NB, SVM, RF)
  6. Evaluation (accuracy, precision, recall, F1)
  7. Confusion Matrix & Feature Importance plots
  8. Predict on new tickets
  9. Save models

Install dependencies:
    pip install scikit-learn pandas numpy matplotlib seaborn nltk joblib
"""

# ═══════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os, re, warnings, random, string
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as mticker
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model            import LogisticRegression
from sklearn.naive_bayes             import MultinomialNB
from sklearn.svm                     import LinearSVC
from sklearn.ensemble                import RandomForestClassifier
from sklearn.preprocessing           import LabelEncoder
from sklearn.metrics                 import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import joblib

# Download NLTK data quietly
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

# ── output folders ─────────────────────────────────────────────────
os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── global style ───────────────────────────────────────────────────
PALETTE   = ["#2563EB", "#F59E0B", "#10B981", "#EF4444",
             "#8B5CF6", "#EC4899", "#06B6D4"]
PRI_COLOR = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
CAT_COLOR = {"Billing":         "#2563EB",
             "Technical Issue": "#EF4444",
             "Account":         "#10B981",
             "General Query":   "#8B5CF6"}

plt.rcParams.update({
    "figure.facecolor" : "#0F172A",   # dark navy background
    "axes.facecolor"   : "#1E293B",
    "axes.edgecolor"   : "#334155",
    "axes.labelcolor"  : "#CBD5E1",
    "axes.titlecolor"  : "#F1F5F9",
    "xtick.color"      : "#94A3B8",
    "ytick.color"      : "#94A3B8",
    "grid.color"       : "#334155",
    "text.color"       : "#F1F5F9",
    "legend.facecolor" : "#1E293B",
    "legend.edgecolor" : "#334155",
    "font.family"      : "monospace",
})

# ═══════════════════════════════════════════════════════════════════
# 1.  DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════

TEMPLATES = {
    "Billing": {
        "High": [
            "I was charged twice for my subscription this month. Please refund immediately.",
            "Unauthorized charge of $AMOUNT appeared on my credit card. This is fraud!",
            "Double billing issue — $AMOUNT deducted twice from my account. Urgent refund needed.",
            "My account has been suspended but I paid on time. I need access restored now.",
            "Payment failed but $AMOUNT was still deducted from my bank. Need immediate refund.",
            "I never authorized this charge of $AMOUNT. Please investigate and reverse it today.",
        ],
        "Medium": [
            "I need an invoice for the payment I made last month.",
            "My coupon code CODE is not being accepted at checkout. Please fix this.",
            "Can you send me a receipt for my last subscription renewal?",
            "I cancelled my subscription but was charged for another month. Please clarify.",
            "How does billing work for mid-cycle plan upgrades?",
            "I want to switch from monthly to annual billing. Will I be charged immediately?",
        ],
        "Low": [
            "Where can I download my past invoices from the dashboard?",
            "What payment methods do you accept — credit card, PayPal, bank transfer?",
            "Is there a student or non-profit discount available for your service?",
            "How do I update my credit card information on file?",
            "Can you explain the difference between the monthly and annual billing plans?",
            "Do you support automatic billing in currencies other than USD?",
        ],
    },
    "Technical Issue": {
        "High": [
            "The entire platform is down. None of our team can access it. Production is blocked!",
            "Critical bug — data is being corrupted when we export reports. Major issue.",
            "Application crashes immediately on launch after today's update. Completely unusable.",
            "Our API integration is returning 500 errors and has broken our entire workflow.",
            "Security alert — we suspect our account has been compromised. Need immediate help.",
            "Server is completely unresponsive. Thousands of users are affected right now.",
        ],
        "Medium": [
            "The export to PDF feature is not working. It just spins and never downloads.",
            "I'm getting a 404 not found error on the analytics page.",
            "The mobile app keeps logging me out every few minutes. Very frustrating.",
            "Charts on the dashboard are not loading. The rest of the page works fine.",
            "Notifications are delayed by several hours. I'm missing important alerts.",
            "The file upload fails for anything over 5 MB with no error message.",
        ],
        "Low": [
            "The dark mode toggle does not save my preference after I log out.",
            "Spell check is not working inside the notes editor.",
            "The search bar is a bit slow to respond. Minor but annoying.",
            "Some icons in the sidebar look blurry on my retina display.",
            "The date picker does not let me type a date manually.",
            "Minor UI alignment issue in the settings panel on mobile.",
        ],
    },
    "Account": {
        "High": [
            "I cannot login. Password reset emails are not arriving in my inbox either.",
            "My account was deleted without warning. I have lost all my project data!",
            "Someone else is logged into my account. I think I have been hacked.",
            "My 2FA device was lost and now I am completely locked out of my account.",
            "Account suspended with no explanation. I need access restored immediately.",
            "I cannot access my account and there is sensitive data inside. Urgent!",
        ],
        "Medium": [
            "I need to transfer my account ownership to a colleague who is taking over.",
            "How do I merge two accounts I created by mistake?",
            "I want to delete my account and all associated data (GDPR request).",
            "Can I change the email address associated with my account?",
            "How do I add a new admin user to our team workspace?",
            "I need to revoke API access for a former employee's integration.",
        ],
        "Low": [
            "How do I update my profile picture and display name?",
            "Where can I change my email notification preferences?",
            "Can I set a custom timezone for my account?",
            "How do I enable two-factor authentication on my account?",
            "I'd like to change my username. Is that possible?",
            "How do I set up SSO for my organization?",
        ],
    },
    "General Query": {
        "High": [
            "We are evaluating your product for enterprise use. Need a demo call today — very urgent.",
            "Our legal team requires HIPAA and GDPR compliance documentation by end of day.",
        ],
        "Medium": [
            "What is the maximum file size I can upload to the platform?",
            "Does your service integrate with Slack and Microsoft Teams?",
            "How many users can I add under the business plan?",
            "Is there an offline mode available for the desktop app?",
            "What happens to my data if I downgrade my subscription plan?",
            "Does your platform support bulk data imports via CSV or API?",
        ],
        "Low": [
            "Do you have a public API I can use to pull my data programmatically?",
            "Where can I find the REST API documentation?",
            "Is there a community forum or Discord channel for users?",
            "Do you offer white-label options for agencies and resellers?",
            "What are your support hours and typical response times?",
            "Is there a changelog or release notes page I can follow?",
        ],
    },
}

FILLERS = [
    " Please help.", " Thank you.", " Looking forward to your response.",
    " This is very important to us.", " Appreciate your quick response.", "",
]

def _fill(text):
    text = text.replace("AMOUNT", str(random.randint(10, 500)))
    text = text.replace("CODE",   f"SAVE{random.randint(10,50)}")
    return text + random.choice(FILLERS)

def generate_dataset(n=2000, seed=42):
    random.seed(seed); np.random.seed(seed)
    cats = list(TEMPLATES.keys())
    pris = ["High", "Medium", "Low"]
    rows = []
    for i in range(n):
        cat = random.choice(cats)
        pri = random.choices(pris, weights=[0.20, 0.45, 0.35])[0]
        pool = TEMPLATES[cat].get(pri, TEMPLATES[cat]["Medium"])
        rows.append({
            "ticket_id"  : f"TKT-{i+1:05d}",
            "ticket_text": _fill(random.choice(pool)),
            "category"   : cat,
            "priority"   : pri,
        })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[DATA]  Generated {len(df):,} tickets")
    print(f"        Categories : {df['category'].value_counts().to_dict()}")
    print(f"        Priorities : {df['priority'].value_counts().to_dict()}")
    return df

# ═══════════════════════════════════════════════════════════════════
# 2.  EDA VISUALIZATIONS  (saved + shown)
# ═══════════════════════════════════════════════════════════════════

def plot_eda(df):
    print("\n[PLOT]  Drawing EDA dashboard …")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("SUPPORT TICKET DATASET — EXPLORATORY ANALYSIS",
                 fontsize=16, fontweight="bold", color="#F1F5F9", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (A) Category bar ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cat_counts = df["category"].value_counts()
    colors = [CAT_COLOR[c] for c in cat_counts.index]
    bars = ax1.barh(cat_counts.index, cat_counts.values, color=colors, height=0.6)
    for bar in bars:
        ax1.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                 f"{int(bar.get_width())}", va="center", fontsize=10, color="#F1F5F9")
    ax1.set_title("Ticket Categories", fontweight="bold")
    ax1.set_xlabel("Count")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.invert_yaxis()
    sns.despine(ax=ax1)

    # ── (B) Priority donut ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pri_counts = df["priority"].value_counts().reindex(["High","Medium","Low"])
    wedge_colors = [PRI_COLOR[p] for p in pri_counts.index]
    wedges, texts, autotexts = ax2.pie(
        pri_counts.values, labels=pri_counts.index,
        autopct="%1.1f%%", colors=wedge_colors,
        startangle=140, pctdistance=0.80,
        wedgeprops=dict(width=0.55, edgecolor="#0F172A", linewidth=2),
    )
    for t in texts:    t.set_color("#CBD5E1"); t.set_fontsize(11)
    for t in autotexts: t.set_color("#0F172A"); t.set_fontsize(10); t.set_fontweight("bold")
    ax2.set_title("Priority Distribution", fontweight="bold")

    # ── (C) Stacked bar: priority within category ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cross = df.groupby(["category","priority"]).size().unstack(fill_value=0)
    cross = cross.reindex(columns=["High","Medium","Low"])
    bottom = np.zeros(len(cross))
    for pri in ["High","Medium","Low"]:
        vals = cross[pri].values
        ax3.bar(cross.index, vals, bottom=bottom,
                color=PRI_COLOR[pri], label=pri, alpha=0.90, edgecolor="#0F172A", linewidth=0.6)
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v > 0:
                ax3.text(i, b + v/2, str(v), ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax3.set_title("Priority Breakdown per Category", fontweight="bold")
    ax3.set_xlabel("Category"); ax3.set_ylabel("Count")
    ax3.legend(title="Priority", loc="upper right")
    ax3.set_xticklabels(cross.index, rotation=15, ha="right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(ax=ax3)

    # ── (D) Text length distribution ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    df["text_len"] = df["ticket_text"].apply(lambda x: len(x.split()))
    for cat, grp in df.groupby("category"):
        ax4.hist(grp["text_len"], bins=20, alpha=0.55,
                 color=CAT_COLOR[cat], label=cat, edgecolor="#0F172A", linewidth=0.4)
    ax4.set_title("Word Count Distribution", fontweight="bold")
    ax4.set_xlabel("Word Count"); ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(ax=ax4)

    # ── (E) Heatmap: avg text length by cat × priority ────────────
    ax5 = fig.add_subplot(gs[1, 1])
    heat_data = df.groupby(["category","priority"])["text_len"].mean().unstack()
    heat_data = heat_data.reindex(columns=["High","Medium","Low"])
    sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="#0F172A", ax=ax5,
                cbar_kws={"label": "Avg Words"})
    ax5.set_title("Avg Word Count (Cat × Priority)", fontweight="bold")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0, fontsize=9)

    # ── (F) Ticket count over synthetic ticket IDs (trend) ────────
    ax6 = fig.add_subplot(gs[1, 2])
    df["ticket_num"] = df["ticket_id"].str.extract(r"(\d+)").astype(int)
    df_sorted = df.sort_values("ticket_num")
    window = 100
    for cat in df["category"].unique():
        mask   = df_sorted["category"] == cat
        series = mask.astype(int).rolling(window, min_periods=1).mean()
        ax6.plot(df_sorted["ticket_num"], series * 100,
                 label=cat, color=CAT_COLOR[cat], linewidth=2)
    ax6.set_title(f"Category Rolling Share (window={window})", fontweight="bold")
    ax6.set_xlabel("Ticket #"); ax6.set_ylabel("Rolling % share")
    ax6.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax6.legend(fontsize=8)
    ax6.grid(linestyle="--", alpha=0.4)
    sns.despine(ax=ax6)

    plt.savefig("plots/01_eda_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("[PLOT]  Saved → plots/01_eda_dashboard.png")

# ═══════════════════════════════════════════════════════════════════
# 3.  TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

class TextPreprocessor:
    def __init__(self):
        self.lemma  = WordNetLemmatizer()
        self.stops  = set(stopwords.words("english")) | {
            "hi","hello","dear","regards","thanks","thank","please",
            "help","need","want","would","could","use","also","get",
            "got","im","ive","cant","dont","ticket","issue","problem",
            "support","team","account",
        }

    def clean(self, text):
        if not isinstance(text, str): text = str(text)
        text = text.lower()
        text = re.sub(r"<[^>]+>",      " ", text)   # HTML
        text = re.sub(r"http\S+",       " ", text)   # URLs
        text = re.sub(r"\S+@\S+",       " ", text)   # emails
        text = re.sub(r"[^a-z0-9\s]",  " ", text)   # punctuation
        text = re.sub(r"\s+",           " ", text).strip()
        tokens = [
            self.lemma.lemmatize(t)
            for t in text.split()
            if t not in self.stops and len(t) > 2
        ]
        return " ".join(tokens)

# ═══════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_models(X_tr, y_tr, X_te, y_te, label_names, task_name):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "Naive Bayes"        : MultinomialNB(alpha=0.1),
        "Linear SVC"         : LinearSVC(C=1.0, max_iter=2000, dual=True),
        "Random Forest"      : RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    results    = {}
    best_model = None
    best_f1    = -1

    print(f"\n{'─'*58}")
    print(f"  {task_name} CLASSIFICATION")
    print(f"{'─'*58}")
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc    = accuracy_score (y_te, y_pred) * 100
        prec   = precision_score(y_te, y_pred, average="macro", zero_division=0) * 100
        rec    = recall_score   (y_te, y_pred, average="macro", zero_division=0) * 100
        f1     = f1_score       (y_te, y_pred, average="macro", zero_division=0) * 100

        results[name] = dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, model=model)
        marker = " ◀ best" if f1 > best_f1 else ""
        print(f"  {name:<22} {acc:>6.1f}% {prec:>6.1f}% {rec:>6.1f}% {f1:>6.1f}%{marker}")

        if f1 > best_f1:
            best_f1    = f1
            best_model = model

    print(f"\n  🏆 Best: {[k for k,v in results.items() if v['model'] is best_model][0]}"
          f"  (F1={best_f1:.1f}%)")
    return results, best_model

# ═══════════════════════════════════════════════════════════════════
# 5.  EVALUATION VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def plot_model_comparison(cat_results, pri_results):
    print("\n[PLOT]  Drawing model comparison …")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("MODEL PERFORMANCE COMPARISON",
                 fontsize=15, fontweight="bold", y=1.01)

    metric_colors = {"Accuracy": PALETTE[0], "Precision": PALETTE[1],
                     "Recall":   PALETTE[2],  "F1":        PALETTE[3]}

    for ax, (results, title) in zip(axes, [
        (cat_results, "Category Classification"),
        (pri_results, "Priority Classification"),
    ]):
        model_names = list(results.keys())
        metrics     = ["Accuracy", "Precision", "Recall", "F1"]
        x = np.arange(len(model_names))
        w = 0.20

        for i, metric in enumerate(metrics):
            vals = [results[m][metric] for m in model_names]
            bars = ax.bar(x + (i - 1.5) * w, vals, w,
                          label=metric, color=metric_colors[metric], alpha=0.88,
                          edgecolor="#0F172A", linewidth=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.8,
                        f"{bar.get_height():.0f}",
                        ha="center", va="bottom", fontsize=7.5, color="#CBD5E1")

        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=10)
        ax.set_ylim(0, 118)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylabel("Score (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("plots/02_model_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("[PLOT]  Saved → plots/02_model_comparison.png")


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    print(f"[PLOT]  Drawing confusion matrix ({title}) …")
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"CONFUSION MATRIX — {title.upper()}",
                 fontsize=14, fontweight="bold")

    for ax, data, fmt, label in zip(
        axes,
        [cm,      cm_pct],
        ["d",     ".1f"],
        ["Count", "% of True Class"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    cmap="Blues" if "Count" in label else "YlOrRd",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#0F172A", ax=ax,
                    cbar_kws={"label": label})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title(label,        fontsize=11, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fname = f"plots/03_confusion_{title.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"[PLOT]  Saved → {fname}")


def plot_top_features(model, tfidf, class_names, title):
    """Works for LR, LinearSVC (coef_) or Random Forest (feature_importances_)."""
    feat_names = np.array(tfidf.get_feature_names_out())

    if not (hasattr(model, "coef_") or hasattr(model, "feature_importances_")):
        print("[PLOT]  Model has no coef_ — skipping feature plot.")
        return

    print(f"[PLOT]  Drawing top features ({title}) …")

    if hasattr(model, "coef_"):
        coef      = model.coef_
        if coef.ndim == 1: coef = coef.reshape(1, -1)
        n_classes = coef.shape[0]
        n_cols    = min(n_classes, 4)
        n_rows    = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5.5 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle(f"TOP FEATURES PER CLASS — {title.upper()}",
                     fontsize=13, fontweight="bold")
        N = 12
        for i, cls in enumerate(class_names):
            ax      = axes[i]
            top_idx = np.argsort(coef[i])[::-1][:N]
            words   = feat_names[top_idx][::-1]
            scores  = coef[i][top_idx][::-1]
            color   = PALETTE[i % len(PALETTE)]
            ax.barh(range(N), scores, color=color, alpha=0.85, edgecolor="#0F172A")
            ax.set_yticks(range(N)); ax.set_yticklabels(words, fontsize=9)
            ax.set_title(cls, fontweight="bold", fontsize=11)
            ax.set_xlabel("Coefficient")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            sns.despine(ax=ax)
        for j in range(i + 1, len(axes)): axes[j].set_visible(False)

    else:  # RandomForest
        imp     = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.suptitle(f"TOP 20 FEATURES — {title.upper()}",
                     fontsize=13, fontweight="bold")
        ax.barh(range(20), imp[top_idx][::-1], color=PALETTE[0], alpha=0.85)
        ax.set_yticks(range(20))
        ax.set_yticklabels(feat_names[top_idx][::-1], fontsize=9)
        ax.set_xlabel("Importance")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"plots/04_features_{title.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"[PLOT]  Saved → {fname}")


def plot_cross_val(X, y_cat, y_pri, cat_enc, pri_enc):
    """5-fold cross-validation scores for best models."""
    print("\n[PLOT]  Drawing cross-validation scores …")
    lr  = LogisticRegression(max_iter=1000, solver="lbfgs")
    svc = LinearSVC(C=1.0, max_iter=2000, dual=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("5-FOLD CROSS-VALIDATION — F1 MACRO",
                 fontsize=14, fontweight="bold")

    for ax, (model, y, task) in zip(axes, [
        (lr,  y_cat, "Category"),
        (svc, y_pri, "Priority"),
    ]):
        scores = cross_val_score(model, X, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="f1_macro")
        folds  = [f"Fold {i+1}" for i in range(5)]
        bars   = ax.bar(folds, scores * 100,
                        color=PALETTE[:5], alpha=0.88,
                        edgecolor="#0F172A", linewidth=0.5)
        ax.axhline(scores.mean() * 100, color="#F59E0B", linewidth=2,
                   linestyle="--", label=f"Mean = {scores.mean()*100:.1f}%")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{task} (LR / LinearSVC)", fontweight="bold")
        ax.set_ylabel("F1 Macro (%)")
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("plots/05_cross_validation.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("[PLOT]  Saved → plots/05_cross_validation.png")


def plot_prediction_summary(predictor_df):
    """Visualise the batch prediction results."""
    print("\n[PLOT]  Drawing prediction summary …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("LIVE PREDICTION SUMMARY", fontsize=14, fontweight="bold")

    # Category distribution
    cat_cnt  = predictor_df["Category"].value_counts()
    cat_clrs = [CAT_COLOR.get(c, PALETTE[0]) for c in cat_cnt.index]
    axes[0].barh(cat_cnt.index, cat_cnt.values, color=cat_clrs, alpha=0.88)
    axes[0].set_title("Predicted Categories", fontweight="bold")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(cat_cnt.values):
        axes[0].text(v + 0.05, i, str(v), va="center")
    sns.despine(ax=axes[0])

    # Priority distribution
    raw_pri  = predictor_df["Priority"].str.replace(r"[🔴🟡🟢] ", "", regex=True)
    pri_cnt  = raw_pri.value_counts().reindex(["High","Medium","Low"]).fillna(0)
    pri_clrs = [PRI_COLOR[p] for p in pri_cnt.index]
    axes[1].bar(pri_cnt.index, pri_cnt.values, color=pri_clrs, alpha=0.88,
                edgecolor="#0F172A", linewidth=0.5)
    axes[1].set_title("Predicted Priorities", fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(pri_cnt.values):
        axes[1].text(i, v + 0.05, str(int(v)), ha="center")
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.savefig("plots/06_prediction_summary.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("[PLOT]  Saved → plots/06_prediction_summary.png")

# ═══════════════════════════════════════════════════════════════════
# 6.  PREDICTOR
# ═══════════════════════════════════════════════════════════════════

PRI_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

def predict_batch(tickets, tfidf, cat_model, pri_model,
                  cat_enc, pri_enc, prep):
    rows = []
    for text in tickets:
        X        = tfidf.transform([prep.clean(text)])
        cat_pred = cat_enc.inverse_transform(cat_model.predict(X))[0]
        pri_pred = pri_enc.inverse_transform(pri_model.predict(X))[0]

        # confidence
        def conf(model, X):
            if hasattr(model, "predict_proba"):
                return f"{model.predict_proba(X).max()*100:.0f}%"
            return "N/A"

        rows.append({
            "Ticket (preview)"   : text[:55] + ("…" if len(text) > 55 else ""),
            "Category"           : cat_pred,
            "Priority"           : f"{PRI_EMOJI.get(pri_pred,'')} {pri_pred}",
            "Cat Conf."          : conf(cat_model, X),
            "Pri Conf."          : conf(pri_model, X),
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    SEP = "\n" + "═"*60

    # ── STEP 1: DATA ──────────────────────────────────────────────
    print(SEP); print("  STEP 1 — GENERATE DATASET")
    df = generate_dataset(n=2000)

    # ── STEP 2: EDA ───────────────────────────────────────────────
    print(SEP); print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    plot_eda(df)

    # ── STEP 3: PREPROCESS ────────────────────────────────────────
    print(SEP); print("  STEP 3 — TEXT PREPROCESSING")
    prep = TextPreprocessor()
    df["clean_text"] = df["ticket_text"].apply(prep.clean)
    print("[PREP]  Sample:")
    print(f"  RAW  : {df['ticket_text'].iloc[0][:80]}")
    print(f"  CLEAN: {df['clean_text'].iloc[0][:80]}")

    # ── STEP 4: FEATURE EXTRACTION ────────────────────────────────
    print(SEP); print("  STEP 4 — TF-IDF FEATURE EXTRACTION")
    cat_enc = LabelEncoder(); pri_enc = LabelEncoder()
    df["cat_enc"] = cat_enc.fit_transform(df["category"])
    df["pri_enc"] = pri_enc.fit_transform(df["priority"])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X     = tfidf.fit_transform(df["clean_text"])
    y_cat = df["cat_enc"];  y_pri = df["pri_enc"]
    print(f"[TFIDF] Matrix shape: {X.shape}")

    X_tr, X_te, yc_tr, yc_te, yp_tr, yp_te = train_test_split(
        X, y_cat, y_pri, test_size=0.20, random_state=42, stratify=y_cat
    )
    print(f"[SPLIT] Train={X_tr.shape[0]}  Test={X_te.shape[0]}")

    # ── STEP 5: TRAIN ─────────────────────────────────────────────
    print(SEP); print("  STEP 5 — TRAIN MODELS")
    cat_results, best_cat = train_models(
        X_tr, yc_tr, X_te, yc_te, cat_enc.classes_, "CATEGORY"
    )
    pri_results, best_pri = train_models(
        X_tr, yp_tr, X_te, yp_te, pri_enc.classes_, "PRIORITY"
    )

    # ── STEP 6: EVALUATE ──────────────────────────────────────────
    print(SEP); print("  STEP 6 — DETAILED EVALUATION")

    yc_pred = best_cat.predict(X_te)
    yp_pred = best_pri.predict(X_te)

    print("\n--- CATEGORY --- Classification Report ---")
    print(classification_report(yc_te, yc_pred, target_names=cat_enc.classes_))
    print("--- PRIORITY --- Classification Report ---")
    print(classification_report(yp_te, yp_pred, target_names=pri_enc.classes_))

    # ── STEP 7: PLOTS ─────────────────────────────────────────────
    print(SEP); print("  STEP 7 — EVALUATION PLOTS")
    plot_model_comparison(cat_results, pri_results)
    plot_confusion_matrix(yc_te, yc_pred, cat_enc.classes_, "Category")
    plot_confusion_matrix(yp_te, yp_pred, pri_enc.classes_, "Priority")
    plot_top_features(best_cat, tfidf, cat_enc.classes_, "Category")
    plot_top_features(best_pri, tfidf, pri_enc.classes_, "Priority")
    plot_cross_val(X, y_cat, y_pri, cat_enc, pri_enc)

    # ── STEP 8: PREDICT ───────────────────────────────────────────
    print(SEP); print("  STEP 8 — PREDICT NEW TICKETS")
    demo_tickets = [
        "I was charged twice this month. Please refund $49 immediately.",
        "Cannot login. Password reset email never arrived.",
        "Application crashes on PDF export — production is blocked!",
        "How do I change my email notification preferences?",
        "Server is completely down. Entire team is blocked. Urgent!",
        "Can I get an invoice for my payment last month?",
        "The dashboard charts are not loading after the update.",
        "How do I transfer my account to a colleague?",
        "What payment methods do you accept?",
        "Unauthorized charge on my credit card. Fraud suspected!",
    ]
    preds = predict_batch(
        demo_tickets, tfidf, best_cat, best_pri, cat_enc, pri_enc, prep
    )
    pd.set_option("display.max_colwidth", 60)
    print(preds.to_string(index=False))
    plot_prediction_summary(preds)

    # ── STEP 9: SAVE MODELS ───────────────────────────────────────
    print(SEP); print("  STEP 9 — SAVE MODELS")
    joblib.dump(tfidf,    "models/tfidf_vectorizer.pkl")
    joblib.dump(best_cat, "models/category_model.pkl")
    joblib.dump(best_pri, "models/priority_model.pkl")
    joblib.dump(cat_enc,  "models/category_encoder.pkl")
    joblib.dump(pri_enc,  "models/priority_encoder.pkl")
    print("[SAVE]  ✅ All models saved to /models/")

    print(SEP)
    print("  ✅  PIPELINE COMPLETE")
    print(f"  📊  6 plots saved in /plots/")
    print(f"  💾  5 model files saved in /models/")
    print(SEP)


if __name__ == "__main__":
    main()
