# Sentence-Level Media Bias Detection

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/NLP-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)

**Comparing TF-IDF + Logistic Regression and fine-tuned BERT for detecting biased language in news sentences.**

**Academic Project | MSc Computer Science (Data Science) | Leiden University**  
**Course:** Text Mining | **Type:** Collaborative course research project

This project investigates whether a contextual Transformer model improves on a conventional lexical baseline in sentence-level media bias classification, particularly when labels are subjective and imbalanced.

**[Read the full research report (PDF)](reports/Text_Mining_Final_Assignment_Report.pdf)**

## Results

The following values are reported in our course report:

| Dataset | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Original | TF-IDF + Logistic Regression | 0.580 | 0.531 |
| Original | Fine-tuned BERT | 0.645 | 0.563 |
| Refined | TF-IDF + Logistic Regression | 0.587 | — |
| Refined | Fine-tuned BERT | 0.786 | — |

*The report provides overall accuracy for the refined runs but does not tabulate their macro-F1 scores in the same summary. Original and refined evaluations use different subsets, so changes in accuracy should not be interpreted as a controlled, same-test-set improvement.*

### Key observations

- **BERT achieved higher overall accuracy** than the TF-IDF baseline on the original data, but the improvement in macro-F1 was smaller because performance remained uneven across classes.
- **Annotation disagreement matters.** Average pairwise Cohen's kappa increased from **0.05** to **0.19** after filtering, although agreement remained low.
- **Model behavior differed under refinement.** The reported BERT accuracy increased substantially on the filtered subset, while Logistic Regression changed little; however, the filtered task also had fewer examples and a different class distribution.
- **Interpretability involves trade-offs.** Logistic Regression coefficients provide a straightforward view of lexical associations, whereas interpreting BERT decisions requires more care; attention-derived scores should not be treated as definitive causal explanations.

## Research question

> To what extent do modern deep learning architectures outperform traditional frequency-based approaches in predictive performance, error patterns, and interpretability for sentence-level media bias detection?

We compare two approaches on the same article-level training/validation split and study how filtering annotations by agreement changes their behavior.

## Dataset

We use the **Sora LREC 2020 Biased Sentences Dataset** (Lim et al., 2020), derived from annotated news articles.

| Dataset | Articles | Sentence-level samples | Unbiased (0) | Biased (1) |
| --- | ---: | ---: | ---: | ---: |
| Original | 46 | 3,925 | 1,053 | 2,872 |
| Refined | 46 | 839 | 149 | 690 |

The scripts expand the source CSV into sentence-level observations using `s0`–`s19` (sentence text), `0`–`19` (annotation scores), and `id_article` (article identifier). Scores of **1** are mapped to **unbiased (0)**; scores of **2–4** to **biased (1)**. Multiple valid annotations may produce multiple sentence-level observations.

We split by **article**, rather than by individual sentence, to avoid placing sentences from the same news article in both training and validation data. The refinement experiment retains observations based on annotation agreement, reducing the dataset size while changing its label distribution.

**Data availability:** The scripts expect a file named `Sora_LREC2020_biasedsentences.csv`. If the source dataset's redistribution terms permit it, place it in the repository; otherwise, obtain it from the original dataset authors and keep it locally. This repository does not claim ownership of the underlying news text or annotations.

## Methods

### 1. TF-IDF + Logistic Regression

- Represent sentences using **unigram and bigram TF-IDF** features.
- Remove English stopwords and infrequent terms.
- Train a **class-weighted Logistic Regression** classifier to address label imbalance.
- Inspect confusion matrices, classification metrics, model coefficients, and high-confidence errors.

### 2. Fine-tuned BERT

- Fine-tune **`bert-base-uncased`** for binary sentence classification.
- Tokenize and pad/truncate inputs to a maximum of **128 tokens**.
- Use class-weighted loss and evaluate with the same article-level partitioning approach.
- Inspect class-wise results and misclassified examples.

### 3. Annotation refinement

Repeat both experiments on a filtered subset based on annotator agreement. This tests how the two approaches behave when ambiguous labels are removed, while also accounting for the resulting reduction in sample size and change in class balance.

## Repository contents

```text
Media-Bias-Detection/
├── README.md
├── requirements.txt                 # Add compatible dependency versions
├── data/                            # Dataset, if redistribution is permitted
│   └── Sora_LREC2020_biasedsentences.csv
├── src/
│   ├── tfidf_lr_baseline.py
│   ├── tfidf_lr_refined.py
│   ├── bert_baseline.py
│   └── bert_refined.py
├── notebooks/
│   ├── bert_baseline.ipynb
│   └── bert_refined.ipynb
└── reports/
    └── Text_Mining_Final_Assignment_Report.pdf
```

The names above are suggested repository filenames; rename the original assignment files accordingly if using this structure.

## Running the experiments

1. Create a Python environment and install the dependencies listed in `requirements.txt` once it has been prepared and tested.
2. Place `Sora_LREC2020_biasedsentences.csv` where the scripts expect it, or update their file paths.
3. Run the TF-IDF experiments as standard Python scripts.
4. Open the BERT notebooks in **Google Colab**, adjust the Google Drive CSV path, and run the cells. The original BERT `.py` exports contain notebook-specific commands (such as `!pip install` and Google Drive mounting), so they need adaptation before running as standalone Python files.

**Reproducibility note:** The source files were developed for coursework and have not yet been consolidated into a fully tested, one-command reproduction pipeline. The report documents the experimental settings and results.

## Academic context and contributions

This project was completed as part of the **Text Mining** course in the **MSc Computer Science (Data Science)** programme at **Leiden University**. It was a collaborative course research project, not an independently published paper.

**Team contributions:**

- **Jian Song:** Implemented the **TF-IDF + Logistic Regression** baseline.
- **Rahul Ray:** Implemented the **fine-tuned BERT** model.
- **Both authors:** Collaborated on the experimental comparison, discussions, data-refinement investigation, and final report.

### Full research report

📄 **[Text Mining Final Assignment Report (PDF)](reports/Text_Mining_Final_Assignment_Report.pdf)**

The nine-page report includes the research question, experimental design, model implementation details, evaluation results, error analysis, discussion, and references. It credits both authors.

## Reference

Lim, S., Jatowt, A., Färber, M., & Yoshikawa, M. (2020). *Annotating and Analyzing Biased Sentences in News Articles Using Crowdsourcing.*

*Academic course project; not a published research paper by the repository authors.*
