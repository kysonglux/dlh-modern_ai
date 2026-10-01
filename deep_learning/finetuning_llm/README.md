---
tags: [llm, nlp, huggingface, transformers, fine-tuning, text-classification]
created: 2026-10-01
status: in-progress
---

# Fine-Tuning Transformers for Text Classification — Learning Notes

> [!info] Purpose
> Notes covering: the Transformer architecture basics, DistilBERT, text classification & sentiment analysis, the mechanics of tokenization (subwords, attention masks, truncation, dynamic padding), fine-tuning with the Hugging Face `Trainer`, evaluation metrics (F1/precision/recall, confusion matrix), inference pipelines, and sharing a model on the Hub.

## Table of Contents
- [[#1. What is a Transformer model?]]
- [[#2. What is DistilBERT?]]
- [[#3. What is text classification?]]
- [[#4. What is sentiment analysis?]]
- [[#5. What is Fine-Tuning?]]
- [[#6. Why is fine-tuning needed?]]
- [[#7. Why is tokenization required?]]
- [[#8. What is subword tokenization?]]
- [[#9. What is the role of attention masks?]]
- [[#10. Why is truncation used?]]
- [[#11. What is dynamic padding?]]
- [[#12. Why compute weighted metrics like F1, precision, and recall?]]
- [[#13. What does a confusion matrix show?]]
- [[#14. What is a Hugging Face inference pipeline used for?]]
- [[#15. How can we push a fine-tuned model to the Hugging Face Hub?]]
- [[#The fine-tuning workflow end-to-end]]
- [[#Glossary]]
- [[#Resources checklist]]

---

## 1. What is a Transformer model?

A **Transformer** is a neural network architecture built around **self-attention**, introduced to process sequences (like text) without needing to read them strictly left-to-right as older RNN/LSTM models did.

Core pieces:
- **Self-attention** — for every token in the input, the model computes a weighted combination of *every other token*, so it can directly relate words that are far apart in the sentence (e.g., a pronoun and the noun it refers to many words earlier).
- **Multi-head attention** — several attention computations run in parallel ("heads"), each potentially capturing a different kind of relationship (syntax, coreference, topic, etc.).
- **Feed-forward layers** — applied to each position independently, adding non-linear transformation capacity after attention mixes information across positions.
- **Positional encodings** — since attention itself has no notion of word order, position information is injected separately.
- **Encoder vs. decoder blocks**:
  - *Encoder-only* models (BERT, DistilBERT, RoBERTa) — read the whole sequence at once (bidirectional), great for *understanding* tasks like classification.
  - *Decoder-only* models (GPT family) — read left-to-right only, great for *generation*.
  - *Encoder-decoder* models (T5, BART) — encode input, then generate output, great for translation/summarization.

For this chapter, the relevant branch is the **encoder-only** family, since text classification needs rich understanding of a whole input, not generation.

---

## 2. What is DistilBERT?

**DistilBERT** is a smaller, faster version of BERT produced through **knowledge distillation**: a large "teacher" model (BERT) is used to train a smaller "student" model (DistilBERT) to mimic its behavior.

Key facts:
- ~40% fewer parameters than BERT-base, runs ~60% faster, while retaining roughly ~97% of BERT's language understanding performance on many benchmarks.
- Still an **encoder-only, bidirectional** Transformer — same basic mechanics as BERT (trained with Masked Language Modeling), just fewer layers and no "next sentence prediction" objective.
- Common base checkpoint: `distilbert-base-uncased` — "uncased" means text is lowercased before tokenization, so it doesn't distinguish "Apple" (company) from "apple" (fruit) by casing alone.
- Popular choice for **text classification / sentiment analysis** fine-tuning because it's cheap to train and run while still being quite accurate — a good balance between compute cost and quality.

---

## 3. What is text classification?

**Text classification** is the task of assigning one (or more) predefined labels/categories to a piece of text.

- **Single-label (multi-class)**: each input gets exactly one label out of several possible classes (e.g., topic categories: sports / politics / tech).
- **Multi-label**: an input can belong to several classes simultaneously (e.g., a support ticket tagged both "billing" and "urgent").
- **Binary classification**: a special case with exactly two classes (e.g., spam vs. not spam).

In the Hugging Face ecosystem, this is implemented by attaching a **classification head** (a simple linear layer) on top of the Transformer's pooled output (typically the representation of the special `[CLS]` token), trained to output a probability distribution over the label set. This is exactly what `AutoModelForSequenceClassification` sets up automatically.

A classic benchmark suite for this kind of task is **GLUE** (General Language Understanding Evaluation) — a collection of diverse text classification/understanding tasks used to compare models.

---

## 4. What is sentiment analysis?

**Sentiment analysis** is a specific, very common form of text classification where the labels represent an emotional tone or opinion polarity, for example:
- Binary: *positive* / *negative*
- Ternary: *positive* / *neutral* / *negative*
- Fine-grained emotion sets: *joy, sadness, anger, fear, surprise, love* (e.g., the `dair-ai/emotion` dataset)

It's used heavily on product reviews, social media/tweets, customer feedback, and support tickets to automatically gauge how people feel about something at scale. Architecturally it's not special — it's just text classification with sentiment-flavored labels, so the same fine-tuning pipeline (tokenize → classification head → train → evaluate) applies.

---

## 5. What is Fine-Tuning?

**Fine-tuning** means taking a model that has already been pretrained on a large, general corpus (e.g., DistilBERT pretrained via masked language modeling on generic text) and **continuing training it on a smaller, task-specific labeled dataset** so it specializes in a particular task (e.g., classifying tweets as positive/negative).

Typical fine-tuning recipe for classification:
1. Load a pretrained base model and its matching tokenizer (`AutoTokenizer`, `AutoModelForSequenceClassification`).
2. Replace/attach a task-specific head (done automatically when you specify `num_labels`).
3. Tokenize the labeled dataset.
4. Train for a small number of epochs with a relatively low learning rate (so you adapt the model without destroying its pretrained knowledge — this is sometimes called avoiding "catastrophic forgetting").
5. Evaluate on a held-out validation/test split.

In Hugging Face `transformers`, this is usually orchestrated with the **`Trainer`** class plus **`TrainingArguments`** (epochs, learning rate, batch size, evaluation strategy, etc.), rather than writing a manual training loop — though a manual loop (as in the "Full training loop" guide) is useful for understanding what `Trainer` does under the hood.

---

## 6. Why is fine-tuning needed?

- A pretrained base model has general language understanding but **no notion of your specific labels** (it has never seen "positive"/"negative" as output categories) — fine-tuning teaches it the mapping from input text to your task's specific output space.
- It's **far more data- and compute-efficient** than training a model from scratch: the model already "knows" grammar, semantics, and world knowledge, so fine-tuning only needs to adjust it for the task, typically with a much smaller labeled dataset and far fewer training steps.
- It lets one general-purpose architecture be **reused across many different tasks and domains** just by swapping the fine-tuning dataset and output head, rather than designing a new model per task.
- It typically **outperforms zero-shot prompting** of the base model on well-defined classification tasks where labeled training data is available, because the model's weights are directly optimized for that exact label set and distribution.

---

## 7. Why is tokenization required?

Neural networks only operate on numbers, not raw characters or words. **Tokenization** is the preprocessing step that converts raw text into a sequence of numeric IDs the model can consume:

`raw text → tokens (sub-word pieces) → token IDs (integers) → embedding lookup → model input`

Tokenization also standardizes input formatting: it adds required special tokens (like `[CLS]` at the start and `[SEP]` at the end for BERT-family models), and produces the matching **attention mask** and (if needed) **token type IDs**. Hugging Face's `AutoTokenizer` automatically loads the exact tokenizer that matches a given pretrained checkpoint, so token IDs line up with what the model was trained on.

---

## 8. What is subword tokenization?

Rather than splitting text into whole words (which creates huge vocabularies and can't handle unseen/rare words) or single characters (which loses efficiency and some meaning), modern tokenizers split words into **subword units** — common algorithms include **WordPiece** (used by BERT/DistilBERT), Byte-Pair Encoding (BPE, used by GPT-2/RoBERTa), and SentencePiece.

How it helps:
- Frequent whole words stay as single tokens (`"the"`, `"running"`).
- Rare or unseen words get broken into familiar pieces (e.g., `"unbelievable"` → `"un"`, `"##believ"`, `"##able"`), so the model never hits a completely unknown word — it can still build meaning from known sub-parts.
- This keeps the **vocabulary size manageable** (e.g., ~30,000 tokens for BERT/DistilBERT) while still covering effectively infinite words, including typos, new terms, and multiple languages to some extent.

---

## 9. What is the role of attention masks?

Because models process inputs in **batches of equal length**, shorter sequences in a batch get padded with filler (`[PAD]`) tokens to match the longest sequence. The **attention mask** is a parallel binary array (same length as the input) that tells the model which positions are *real tokens* (mask = 1) and which are *padding* (mask = 0).

Why it matters: without it, the model's self-attention would treat meaningless padding tokens as real content and let them influence the representations of actual words, corrupting the output. The attention mask ensures padding positions are effectively ignored during the attention computation — the model only "pays attention to" genuine tokens.

---

## 10. Why is truncation used?

Every Transformer model has a **maximum sequence length** (context window) it was trained with and can structurally accept (e.g., 512 tokens for BERT/DistilBERT). **Truncation** cuts input sequences down to that maximum length (keeping, e.g., only the first N tokens, or a chosen truncation strategy) so that:

- Inputs longer than the limit don't cause errors or need to be rejected outright.
- Memory and compute usage stay bounded and predictable — self-attention cost grows roughly quadratically with sequence length, so excessively long inputs are expensive.
- Batches of mixed-length text can be processed uniformly alongside padding and attention masks.

The trade-off is that truncated content beyond the cutoff is simply lost to the model, which is a consideration when working with long documents.

---

## 11. What is dynamic padding?

Instead of padding **every** example in the entire dataset to one fixed global maximum length upfront (which wastes a lot of compute on mostly-padding batches), **dynamic padding** pads each **batch** only to the length of its own longest example, computed on the fly at batch-creation time.

- This is exactly what Hugging Face's **`DataCollatorWithPadding`** does: it's passed to the `Trainer`/`DataLoader` and pads each mini-batch dynamically during training/evaluation, rather than requiring a fixed pre-padded dataset.
- Benefit: significantly less wasted computation on padding tokens overall, especially when sequence lengths vary a lot across the dataset — this speeds up training without changing results.

---

## 12. Why do we compute weighted metrics like F1, precision, and recall?

For classification, raw **accuracy** (percent correct) can be misleading, especially with **imbalanced classes** (e.g., 95% negative, 5% positive reviews) — a model that always predicts "negative" would score 95% accuracy while being useless.

- **Precision** — of everything the model predicted as class X, how much was actually class X? (Measures false-positive control.)
- **Recall** — of everything that actually was class X, how much did the model correctly find? (Measures false-negative control.)
- **F1 score** — the harmonic mean of precision and recall, balancing both into a single number; it penalizes models that sacrifice one for the other.

Because real datasets usually have **more than two classes** and/or **class imbalance**, these per-class scores need to be combined into one overall number, which is where averaging strategies come in:
- **Macro average** — compute the metric per class, then average the classes equally (treats rare classes as equally important as common ones).
- **Micro average** — aggregate contributions (true/false positives/negatives) across all classes first, then compute the metric once (dominated by common classes).
- **Weighted average** — like macro, but weights each class's score by how many true instances it has (accounts for imbalance while still reporting per-class performance).

Hugging Face's `Trainer` uses a `compute_metrics` function (receiving an **`EvalPrediction`** object with predictions and true labels) typically built with scikit-learn's `precision_recall_fscore_support` and `accuracy_score`, so you can report accuracy alongside weighted/macro F1, precision, and recall.

---

## 13. What does a confusion matrix show?

A **confusion matrix** is a table that breaks down a classifier's predictions against the true labels, row = actual class, column = predicted class (or vice versa depending on convention). For binary classification it has four cells:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

It shows **exactly where the model is going wrong**, not just an aggregate score: e.g., it reveals if the model systematically confuses two specific classes, or if errors are concentrated as false positives vs. false negatives. Precision, recall, and F1 can all be derived directly from the confusion matrix's cell counts, and for multi-class problems the matrix extends to an N×N grid showing confusion between every pair of classes.

---

## 14. What is a Hugging Face inference pipeline used for?

The **`pipeline()`** function wraps the full inference flow — tokenizer → model forward pass → post-processing (e.g., turning logits into human-readable labels with confidence scores) — into a single, simple call. For a fine-tuned text classification model:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/your-fine-tuned-model")
classifier("This movie was absolutely wonderful!")
# -> [{'label': 'POSITIVE', 'score': 0.998}]
```

It's used for:
- **Quick, production-style inference** without manually handling tokenization, tensors, or decoding logits.
- **Prototyping and demoing** a model's behavior immediately after training.
- **Consistent interface** across many task types (classification, generation, translation, etc.) — swapping the task name and model is often all that's needed.

---

## 15. How can we push a fine-tuned model to the Hugging Face Hub?

Once a model is fine-tuned, it can be uploaded (shared) to the **Hugging Face Hub** so others (or your own other projects/environments) can load it directly by its repo name, same as any official model.

Common approaches:
- **Automatic during training**: set `push_to_hub=True` in `TrainingArguments`; the `Trainer` will periodically push checkpoints and the final model to the Hub repo during/after training.
- **Manual, after training**: call `model.push_to_hub("your-username/model-name")` and `tokenizer.push_to_hub("your-username/model-name")` (the tokenizer should be pushed too, so the repo is self-contained and loadable with `AutoTokenizer`/`AutoModel`).
- **CLI**: log in via `huggingface-cli login` (or `notebook_login()` in a notebook) first to authenticate, since pushing requires a valid Hub access token.

Once pushed, anyone (or you, elsewhere) can load it with:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")
model = AutoModelForSequenceClassification.from_pretrained("your-username/model-name")
```
This is the same mechanism that makes every pretrained checkpoint (DistilBERT, RoBERTa, etc.) available in the first place — sharing a fine-tuned model just adds your repo to that same ecosystem.

---

## The fine-tuning workflow end-to-end

```mermaid
flowchart LR
    A[Load dataset] --> B[AutoTokenizer: tokenize + truncate]
    B --> C[DataCollatorWithPadding: dynamic padding per batch]
    C --> D[AutoModelForSequenceClassification: pretrained + new head]
    D --> E[TrainingArguments + Trainer]
    E --> F[compute_metrics: accuracy, F1, precision, recall]
    F --> G[Evaluate on validation/test set]
    G --> H[push_to_hub]
    H --> I[pipeline for inference]
```

1. **Load a dataset** from the Hugging Face Hub (e.g., `dair-ai/emotion`) using the `datasets` library.
2. **Tokenize** with `AutoTokenizer`, applying truncation to respect the model's max length.
3. Use **`DataCollatorWithPadding`** to pad dynamically per batch instead of globally.
4. Load the base model with **`AutoModelForSequenceClassification`**, specifying `num_labels` for the task.
5. Configure **`TrainingArguments`** (epochs, learning rate, batch size, eval strategy, `push_to_hub`).
6. Define a **`compute_metrics`** function using `EvalPrediction` + scikit-learn (`accuracy_score`, `precision_recall_fscore_support` with `average="weighted"`).
7. Train with **`Trainer`**, which handles the training loop, evaluation, and logging.
8. Evaluate — check accuracy, weighted F1/precision/recall, and inspect a confusion matrix for class-level error patterns.
9. **Push the fine-tuned model** (and tokenizer) to the Hub.
10. Use **`pipeline()`** to run quick, real-world inference with the finished model.

---

## Glossary

- **Transformer** — attention-based neural network architecture for sequence data.
- **Encoder-only model** — bidirectional Transformer (e.g., BERT, DistilBERT) suited to understanding/classification tasks.
- **Knowledge distillation** — training a smaller "student" model to mimic a larger "teacher" (how DistilBERT is created from BERT).
- **Classification head** — a small output layer added on top of a pretrained model for a specific label set.
- **Subword tokenization** — splitting words into frequent sub-word pieces (WordPiece/BPE) to balance vocabulary size and coverage.
- **Attention mask** — binary array marking real tokens vs. padding so the model ignores padding during attention.
- **Truncation** — cutting sequences down to a model's max input length.
- **Dynamic padding** — padding per-batch to that batch's longest sequence, instead of a fixed global length.
- **Precision / Recall / F1** — classification quality metrics; F1 balances precision and recall.
- **Macro / Micro / Weighted average** — strategies for combining per-class metrics into one score, especially under class imbalance.
- **Confusion matrix** — table of actual vs. predicted labels showing where a classifier makes mistakes.
- **`Trainer` / `TrainingArguments`** — Hugging Face's high-level training loop and its configuration object.
- **`EvalPrediction`** — object holding model predictions and true labels, passed to a custom `compute_metrics` function.
- **Hugging Face Hub** — the hosted repository for sharing and loading models, datasets, and tokenizers.

---

## Resources checklist
Titles kept for reference (links are intranet-only).

- [ ] What are large language models (LLMs)?
- [ ] What is Hugging Face?
- [ ] Introduction to Hugging Face Transformers
- [ ] How To Get Started With Hugging Face Models?
- [ ] Sequence Classification Using Hugging Face Transformers Library
- [ ] How To Load A Pre-trained Model From Hugging Face?
- [ ] Text Classification on GLUE
- [ ] Transformers and its Trainer
- [ ] F1 Score in Machine Learning
- [ ] Micro, Macro & Weighted Averages of F1 Score
- [ ] Fine-tuning a pretrained model: Introduction
- [ ] Fine-tuning a pretrained model: Processing the data
- [ ] Fine-tuning a pretrained model: Fine-tuning with the Trainer API
- [ ] Fine-tuning a pretrained model: A full training loop
- [ ] Fine-tuning a pretrained model: Understanding Learning Curves

---

## Related notes
- [[LLMs - Learning Notes]]
- [[Transformers Architecture]]
- [[Hugging Face Pipelines Cheatsheet]]
- [[Evaluation Metrics for Classification]]