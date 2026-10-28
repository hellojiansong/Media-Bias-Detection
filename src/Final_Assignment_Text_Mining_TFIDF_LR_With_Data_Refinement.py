import pandas as pd
import numpy as np
import re
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

#2. REFINED PREPROCESSING & CONSENSUS
raw_rows = []
for _, row in df_raw.iterrows():
    article_id = row["id_article"]
    for i in range(20):
        sentence = row[f"s{i}"]
        score = row[str(i)]
        if pd.notna(sentence) and pd.notna(score):
            clean_sentence = re.sub(r'^\[\d+\]:\s*', '', str(sentence)).strip()
            label = 0 if int(score) <= 1 else 1
            raw_rows.append({
                "article_id": article_id,
                "sentence": clean_sentence,
                "label": label
            })

temp_df = pd.DataFrame(raw_rows)

# Majority Voting to find consensus
refined_rows = []
grouped = temp_df.groupby(["article_id", "sentence"])
for (article_id, sentence), group in grouped:
    labels = group["label"].tolist()
    majority_label = max(set(labels), key=labels.count)
    agreement = labels.count(majority_label) / len(labels)
    if len(labels) >= 1 and agreement >= 0.5:  # Inclusive threshold
        refined_rows.append({
            "article_id": article_id,
            "sentence": sentence,
            "label": majority_label,
            "agreement": agreement
        })

sent_df = pd.DataFrame(refined_rows)


#2b. INTER-RATER RELIABILITY ON REFINED DATA (COHEN'S KAPPA)
def calculate_refined_kappa(refined_df, raw_temp_df):
    """
    Calculates the Average Pairwise Cohen's Kappa for the subset of
    sentences that passed the consensus filtering.
    """
    # 1. Filter raw ratings to only include sentences present in the refined dataset
    merged_data = pd.merge(
        raw_temp_df,
        refined_df[['article_id', 'sentence']],
        on=['article_id', 'sentence'],
        how='inner'
    )

    # 2. Assign a Rater ID (0, 1, 2...) for each annotation per sentence
    merged_data['rater_id'] = merged_data.groupby(['article_id', 'sentence']).cumcount()

    # 3. Pivot into a Matrix (Rows: Sentences, Columns: Raters)
    kappa_matrix = merged_data.pivot_table(
        index=['article_id', 'sentence'],
        columns='rater_id',
        values='label'
    )

    # 4. Compute Pairwise Kappa for all rater combinations
    raters = kappa_matrix.columns
    pairs = list(itertools.combinations(raters, 2))
    kappas = []

    for r1, r2 in pairs:
        # Find rows where both raters provided a label
        mask = kappa_matrix[r1].notna() & kappa_matrix[r2].notna()
        y1, y2 = kappa_matrix.loc[mask, r1], kappa_matrix.loc[mask, r2]

        # Calculate kappa if there are overlapping items
        if len(y1) > 1:
            k = cohen_kappa_score(y1, y2)
            # Filter out NaN values (happens when a rater has zero variance in labels)
            if not np.isnan(k):
                kappas.append(k)

    # Return the average of all valid pairwise kappa scores
    return np.mean(kappas) if kappas else 0


# Calculate Kappa for the filtered data
avg_kappa = calculate_refined_kappa(sent_df, temp_df)

print("\n" + "=" * 55)
print(f"INTER-RATER RELIABILITY (AFTER MAJORITY VOTING)")
print("-" * 55)
print(f"Average Pairwise Cohen's Kappa: {avg_kappa:.4f}")
print(f"Total Sentences in Refined Set: {len(sent_df)}")
print("Interpretation: < 0.20 = Slight, 0.21-0.40 = Fair Agreement")
print("=" * 55)

#3. ARTICLE-BASED SPLITTING
article_ids = sent_df["article_id"].unique()
train_ids, val_ids = train_test_split(article_ids, test_size=0.2, random_state=42)
train_df = sent_df[sent_df["article_id"].isin(train_ids)].reset_index(drop=True)
val_df = sent_df[sent_df["article_id"].isin(val_ids)].reset_index(drop=True)

#3a. GOLD LABEL DISTRIBUTION OUTPUT
def print_label_distribution(df, name="Dataset"):
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    percents = (counts / total * 100).round(2)
    print(f"\n--- {name} Label Distribution ---")
    print(f"Total sentences: {total}")
    for label, count in counts.items():
        cls_name = "Unbiased" if label == 0 else "Biased"
        print(f"{cls_name} ({label}): {count} ({percents[label]}%)")

print_label_distribution(sent_df, "Full Dataset")
print_label_distribution(train_df, "Training Set")
print_label_distribution(val_df, "Validation Set")

print(f"\nTraining on {len(train_df)} sentences from {len(train_ids)} articles")
print(f"Validating on {len(val_df)} sentences from {len(val_ids)} articles")

#4. FEATURE EXTRACTION (TF-IDF)
tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=5)
X_train = tfidf.fit_transform(train_df["sentence"])
X_val = tfidf.transform(val_df["sentence"])
y_train = train_df["label"]
y_val = val_df["label"]

#5. MODEL TRAINING
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#6. EVALUATION
y_pred = model.predict(X_val)
y_probs = model.predict_proba(X_val)[:, 1]

print("\n" + "=" * 40)
print("TF-IDF + LR CLASSIFICATION REPORT (REFINED)")
print("=" * 40)
print(classification_report(y_val, y_pred, target_names=["Unbiased", "Biased"], digits=3))

#7. ERROR ANALYSIS (TOP 5 CONFIDENT ERRORS)
val_analysis = val_df.copy()
val_analysis["pred"] = y_pred
val_analysis["confidence"] = np.where(y_pred == 1, y_probs, 1 - y_probs)
val_analysis["correct"] = (y_val == y_pred)

print("\n--- TOP 5 MOST CONFIDENT ERRORS ---")
errors = val_analysis[~val_analysis["correct"]].sort_values("confidence", ascending=False)
print(errors[["sentence", "label", "pred", "confidence"]].head(5))

#8. FEATURE IMPORTANCE
feature_names = tfidf.get_feature_names_out()
importance = pd.DataFrame({"word": feature_names, "weight": model.coef_[0]})
print("\n--- TOP 10 BIAS INDICATORS ---")
print(importance.sort_values(by="weight", ascending=False).head(10))

#9. SAVE RESULTS (CONFUSION MATRIX)
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_val, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(cm, display_labels=["Unbiased", "Biased"])
disp.plot(cmap="Blues", ax=ax)
plt.title("Normalized Confusion Matrix (Refined TF-IDF)")
plt.savefig("refined_tfidf_cm.png", dpi=300)
plt.show()