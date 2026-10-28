import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import itertools

#1. DATA LOADING
file_path = "Sora_LREC2020_biasedsentences.csv"
df_raw = pd.read_csv(file_path)

#2. PREPROCESSING & LABEL ALIGNMENT
rows = []
for _, row in df_raw.iterrows():
    for i in range(20):
        sentence = row[f"s{i}"]
        score = row[str(i)]

        if pd.notna(sentence) and pd.notna(score):
            score = int(score)
            label = 0 if score <= 1 else 1  # gold label
            rows.append({
                "article_id": row["id_article"],
                "sentence": str(sentence),
                "label": label
            })

sent_df = pd.DataFrame(rows)
print(f"Total processed sentences: {len(sent_df)}")

#2a. GOLD LABEL DISTRIBUTION (FULL DATASET)
total_sentences = len(sent_df)
total_unbiased = (sent_df["label"] == 0).sum()
total_biased = (sent_df["label"] == 1).sum()
percent_unbiased = total_unbiased / total_sentences * 100
percent_biased = total_biased / total_sentences * 100

print("\n--- Gold Label Distribution (Full Dataset) ---")
print(f"Total sentences: {total_sentences}")
print(f"Unbiased (0): {total_unbiased} ({percent_unbiased:.2f}%)")
print(f"Biased (1): {total_biased} ({percent_biased:.2f}%)")

#2b. INTER-RATER RELIABILITY (COHEN'S KAPPA)
def calculate_inter_rater_reliability(df_raw):
    """
    Calculates the Average Pairwise Cohen's Kappa to assess
    the agreement between different annotators.
    """
    kappa_rows = []
    # Identify unique articles to group raters belonging to the same text
    for article_id in df_raw['id_article'].unique():
        # Get all rows (annotators) for this specific article
        article_group = df_raw[df_raw['id_article'] == article_id].reset_index()

        for rater_idx, row in article_group.iterrows():
            # Process each of the 20 sentence slots
            for i in range(20):
                score = row[str(i)]
                if pd.notna(score):
                    # Align with the binary logic used in your classification task
                    # Label 0: Unbiased (Score <= 1), Label 1: Biased (Score > 1)
                    label = 0 if int(score) <= 1 else 1
                    kappa_rows.append({
                        'item_id': f"{article_id}_s{i}",  # Unique ID for each sentence
                        'rater_id': rater_idx,  # Local rater index for the article
                        'label': label
                    })

    # Create a pivot table where rows are sentences and columns are raters
    kappa_df = pd.DataFrame(kappa_rows)
    pivot_df = kappa_df.pivot(index='item_id', columns='rater_id', values='label')

    # Calculate Cohen's Kappa for all possible pairs of raters
    raters = pivot_df.columns
    pairs = list(itertools.combinations(raters, 2))
    kappas = []

    for r1, r2 in pairs:
        # Extract rows where both raters provided a score
        mask = pivot_df[r1].notna() & pivot_df[r2].notna()
        y1, y2 = pivot_df.loc[mask, r1], pivot_df.loc[mask, r2]

        if len(y1) > 0:
            k = cohen_kappa_score(y1, y2)
            kappas.append(k)

    # Return the mean of all pairwise kappa scores
    return np.mean(kappas)

# Execute the reliability calculation
avg_kappa = calculate_inter_rater_reliability(df_raw)

print("\n" + "=" * 40)
print(f"Average Pairwise Cohen's Kappa: {avg_kappa:.4f}")
print("=" * 40)

#3. ARTICLE-BASED SPLITTING
article_ids = sent_df["article_id"].unique()
train_ids, val_ids = train_test_split(
    article_ids, test_size=0.2, random_state=42
)

train_df = sent_df[sent_df["article_id"].isin(train_ids)].reset_index(drop=True)
val_df = sent_df[sent_df["article_id"].isin(val_ids)].reset_index(drop=True)

print(f"\nTraining on {len(train_df)} sentences from {len(train_ids)} articles")
print(f"Validating on {len(val_df)} sentences from {len(val_ids)} articles")

#3a. GOLD LABEL DISTRIBUTION (TRAIN/VALIDATION)
def print_label_distribution(df, name):
    total = len(df)
    unbiased = (df["label"] == 0).sum()
    biased = (df["label"] == 1).sum()
    print(f"\n{name} label distribution:")
    print(f"Total sentences: {total}")
    print(f"Unbiased: {unbiased} ({unbiased / total * 100:.2f}%)")
    print(f"Biased: {biased} ({biased / total * 100:.2f}%)")

print_label_distribution(train_df, "Training set")
print_label_distribution(val_df, "Validation set")

#4. FEATURE EXTRACTION (TF-IDF)
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)
X_train = tfidf.fit_transform(train_df["sentence"])
X_val = tfidf.transform(val_df["sentence"])
y_train = train_df["label"]
y_val = val_df["label"]

#5. MODEL TRAINING
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

#6. EVALUATION
y_pred = model.predict(X_val)
y_probs = model.predict_proba(X_val)[:, 1]  # probability for Biased class

print("\n" + "=" * 30)
print("TF-IDF + LR CLASSIFICATION REPORT")
print("=" * 30)
print(classification_report(y_val, y_pred, target_names=["Unbiased 0", "Biased 1"], digits=3))

#7. CONFUSION MATRIX
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(cm, display_labels=["Unbiased", "Biased"])
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Normalized Confusion Matrix (TF-IDF + LR)")
plt.savefig(
    "tfidf_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#8. ERROR & CONFIDENCE ANALYSIS
val_analysis_df = val_df.copy()
val_analysis_df["true_label"] = y_val
val_analysis_df["pred_label"] = y_pred
val_analysis_df["confidence"] = np.where(y_pred == 1, y_probs, 1 - y_probs)
val_analysis_df["correct"] = (y_val == y_pred)

print("\n--- TOP 5 MOST CONFIDENT ERRORS ---")
confident_errors = val_analysis_df[~val_analysis_df["correct"]].sort_values("confidence", ascending=False)
print(confident_errors[["sentence", "true_label", "pred_label", "confidence"]].head())


confident_errors_unique = confident_errors.drop_duplicates(subset=["sentence"])
top5_confident_errors = confident_errors_unique.head(5)
top5_confident_errors.to_csv("top5_confident_errors.csv", index=False)
print("Top 5 confident errors saved to top5_confident_errors.csv")

#9. FEATURE IMPORTANCE
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]
importance_df = pd.DataFrame({"word": feature_names, "weight": coefficients})

print("\n--- TOP 10 INDICATORS FOR BIAS ---")
print(importance_df.sort_values(by="weight", ascending=False).head(10))