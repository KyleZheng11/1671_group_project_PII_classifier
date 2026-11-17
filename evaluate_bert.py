"""
evaluate_bert.py -- evaluate the model trained by classifier.py on the Nemotron-PII test split.

It uses the SAME preprocessing as training (imported from classifier.py): the same regex words,
the same overlap handling, the same BIO tags, and the same 512-token windows with stride.
Long documents are split into windows, each window is predicted, and the window predictions are
stitched back into ONE prediction per word, so scores are per document and nothing is cut off.

Run: python evaluate_bert.py
"""

import os
import random

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report as token_report, confusion_matrix
from seqeval.metrics import classification_report as span_report
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Shared settings and the exact same labeling function used for training.
from classifier import CATEGORIES, MAX_LEN, SAVE_PATH, STRIDE, create_bio_labels

MODEL_PATH = SAVE_PATH   # classifier.py saves here, so the two files can't disagree
# How many shuffled test documents to score. Override with:  N_DEV=200 python evaluate_bert.py
# N_DEV=0 scores the whole test split (slow).
N_DEV = int(os.environ.get("N_DEV", 1000))
N_SAMPLES = 10           # how many false positives / false negatives to print
WINDOW_BATCH = 16        # how many windows of one document go through the model at once


def stitch_window_predictions(word_ids_per_window, window_lengths, pred_ids, n_words, id2label):
    """Combine predictions from several overlapping windows into ONE label per word.

    A word in the overlap of two windows has two predictions. We keep the one from the window
    where the word is furthest from the window's edge, because BERT has the most context there.
    Words never seen by any window (which should not happen with windowing) stay "O".
    """
    word_preds = ["O"] * n_words
    best = [-1] * n_words        # best "distance from a window edge" found so far for each word

    for w, word_ids in enumerate(word_ids_per_window):
        last = window_lengths[w] - 1   # index of the last REAL token (padding doesn't count)
        previous = None
        for pos, word in enumerate(word_ids):
            # Only the first subword of each word carries the prediction (matches training).
            if word is not None and word != previous:
                distance_from_edge = min(pos, last - pos)
                if distance_from_edge > best[word]:
                    best[word] = distance_from_edge
                    word_preds[word] = id2label[int(pred_ids[w][pos])]
            previous = word
    return word_preds


def predict(tokens, tokenizer, model):
    """Predict one BIO label for every word of one document."""
    if not tokens:
        return []

    # Same windowing settings as training. padding=True makes all windows the same length
    # so they can be run through the model together.
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    encoded.pop("overflow_to_sample_mapping")  # not an input the model understands

    n_windows = encoded["input_ids"].shape[0]
    word_ids_per_window = [encoded.word_ids(batch_index=w) for w in range(n_windows)]
    window_lengths = encoded["attention_mask"].sum(dim=1).tolist()  # real (unpadded) length of each window

    # Run the windows through BERT in small chunks and keep the winning label id per token.
    pred_ids = []
    with torch.no_grad():
        for i in range(0, n_windows, WINDOW_BATCH):
            chunk = {k: v[i:i + WINDOW_BATCH].to(model.device) for k, v in encoded.items()}
            logits = model(**chunk).logits
            pred_ids.append(logits.argmax(-1).cpu())
    pred_ids = torch.cat(pred_ids).numpy()

    return stitch_window_predictions(
        word_ids_per_window, window_lengths, pred_ids, len(tokens), model.config.id2label
    )


def show(item, tokens):
    label, start, end = item
    span = " ".join(tokens[start:end + 1])
    context = " ".join(tokens[max(0, start - 6):end + 7])
    print(f"  [{label}] '{span}'\n      context: ...{context}...")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # use the GPU if there is one
    model.eval()  # turns off dropout so predictions are deterministic

    # Load the test split and build gold labels with the SAME function used for training.
    data = load_dataset("nvidia/Nemotron-PII")
    test_data = data["test"].select_columns(["text", "spans"]).shuffle(seed=42)
    if N_DEV:
        test_data = test_data.select(range(N_DEV))
    test_data = test_data.map(create_bio_labels)  # adds `tokens` and `bio_labels` columns

    documents = [{"tokens": row["tokens"], "bio_labels": row["bio_labels"]} for row in test_data]

    gold_labels = [doc["bio_labels"] for doc in documents]
    gold_item_count = sum(1 for labels in gold_labels for label in labels if label.startswith("B-"))
    print(f"Documents tested: {len(documents)}   Gold PII items: {gold_item_count}")

    # Predict every document (windowing + stitching happens inside predict()).
    predicted_labels = []
    for i, document in enumerate(documents):
        predicted_labels.append(predict(document["tokens"], tokenizer, model))

    # Token-level metrics ignore B-/I- (so "B-email" and "I-email" both count as "email").
    flat_gold = [label if label == "O" else label[2:] for labels in gold_labels for label in labels]
    flat_predictions = [label if label == "O" else label[2:] for labels in predicted_labels for label in labels]

    print("\n=== TOKEN LEVEL (per category) ===")
    print(token_report(flat_gold, flat_predictions, labels=CATEGORIES, digits=4, zero_division=0))

    # Span-level metrics: a span only counts if its type AND exact boundaries match.
    print("=== SPAN LEVEL (exact match, per category) ===")
    print(span_report(gold_labels, predicted_labels, digits=4, zero_division=0))

    # Confusion matrix: rows are gold labels, columns are predicted labels.
    matrix_labels = CATEGORIES + ["O"]
    short_names = ["first", "last", "email", "street", "phone", "dob", "company", "other", "none"]
    matrix = confusion_matrix(flat_gold, flat_predictions, labels=matrix_labels)

    print("=== CONFUSION MATRIX (token level; rows = gold, columns = predicted) ===")
    print(f"{'':>9}" + "".join(f"{name:>9}" for name in short_names))
    for name, row in zip(short_names, matrix):
        print(f"{name:>9}" + "".join(f"{count:>9}" for count in row))

    # Pick the shortest document with at least three PII items as a readable example.
    example_index, shortest_length = 0, None
    for i, document in enumerate(documents):
        item_count = sum(1 for label in gold_labels[i] if label.startswith("B-"))
        if item_count >= 3 and (shortest_length is None or len(document["tokens"]) < shortest_length):
            example_index, shortest_length = i, len(document["tokens"])

    print("\n=== EXAMPLE INPUT ===")
    print(test_data[example_index]["text"])
    print("\n=== EXAMPLE OUTPUT ===")
    print(f"{'token':22}{'gold':20}predicted")
    for token, gold, prediction in zip(
        documents[example_index]["tokens"], gold_labels[example_index], predicted_labels[example_index]
    ):
        mark = "   <-- wrong" if gold != prediction else ""
        print(f"{token:22}{gold:20}{prediction}{mark}")

    # Compare exact spans to find false positives (predicted but not gold)
    # and false negatives (gold but not predicted).
    false_positives, false_negatives = [], []
    for document, gold, predictions in zip(documents, gold_labels, predicted_labels):
        gold_items = set(get_entities(gold))
        predicted_items = set(get_entities(predictions))
        for item in predicted_items - gold_items:
            false_positives.append((item, document["tokens"]))
        for item in gold_items - predicted_items:
            false_negatives.append((item, document["tokens"]))

    random.seed(42)

    print(f"\n=== FALSE POSITIVES ({len(false_positives)} total, showing up to {N_SAMPLES}) ===")
    for item, tokens in random.sample(false_positives, min(N_SAMPLES, len(false_positives))):
        show(item, tokens)

    print(f"\n=== FALSE NEGATIVES ({len(false_negatives)} total, showing up to {N_SAMPLES}) ===")
    for item, tokens in random.sample(false_negatives, min(N_SAMPLES, len(false_negatives))):
        show(item, tokens)


if __name__ == "__main__":
    main()
