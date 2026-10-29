import random

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report as token_report, confusion_matrix
from seqeval.metrics import classification_report as span_report
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer, AutoModelForTokenClassification

from classifier import create_bio_labels


MODEL_PATH = "./bert_pii_model_full"
N_DEV = 100
N_SAMPLES = 10
CATEGORIES = [
    "first_name",
    "last_name",
    "email",
    "street_address",
    "phone_number",
    "date_of_birth",
    "company_name",
    "other",
]


def predict(tokens, tokenizer, model):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = encoded.to(model.device)

    with torch.no_grad():
        output = model(**encoded)
        predicted_ids = output.logits[0].argmax(-1).tolist()

    predictions = ["O"] * len(tokens)
    word_ids = encoded.word_ids()
    previous_word = None

    # Only use the first subword prediction for each original token.
    for i in range(len(word_ids)):
        word = word_ids[i]

        if word is not None and word != previous_word:
            label_id = predicted_ids[i]
            predictions[word] = model.config.id2label[label_id]

        previous_word = word

    return predictions


def show(item, tokens):
    label, start, end = item
    span = " ".join(tokens[start:end + 1])
    context = " ".join(tokens[max(0, start - 6):end + 7])
    print(f"  [{label}] '{span}'\n      context: ...{context}...")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    print("Model and tokenizer")
    print("Number of labels:", model.config.num_labels)
    print("Labels:", model.config.id2label)

    # Load and label the test documents.
    data = load_dataset("nvidia/Nemotron-PII")
    test_data = data["test"].select_columns(["text", "spans"])
    test_data = test_data.shuffle(seed=42)
    test_data = test_data.select(range(N_DEV))
    documents = create_bio_labels(test_data)

    gold_labels = []
    gold_item_count = 0

    for document in documents:
        labels = document["bio_labels"]
        gold_labels.append(labels)

        for label in labels:
            if label.startswith("B-"):
                gold_item_count += 1

    print(f"Documents tested: {len(documents)}   Gold PII items: {gold_item_count}")

    # Predict the labels for each document.
    predicted_labels = []

    for i in range(len(documents)):
        tokens = documents[i]["tokens"]
        predictions = predict(tokens, tokenizer, model)
        predicted_labels.append(predictions)

        if (i + 1) % 25 == 0:
            print(f"Predicted {i + 1}/{len(documents)}", flush=True)

    # Remove B- and I- prefixes for the token-level metrics.
    flat_gold = []
    flat_predictions = []

    for labels in gold_labels:
        for label in labels:
            if label == "O":
                flat_gold.append("O")
            else:
                flat_gold.append(label[2:])

    for labels in predicted_labels:
        for label in labels:
            if label == "O":
                flat_predictions.append("O")
            else:
                flat_predictions.append(label[2:])

    print("\n=== TOKEN LEVEL (per category) ===")
    print(token_report(
        flat_gold,
        flat_predictions,
        labels=CATEGORIES,
        digits=4,
        zero_division=0,
    ))

    print("=== SPAN LEVEL (exact match, per category) ===")
    print(span_report(gold_labels, predicted_labels, digits=4, zero_division=0))

    # Confusion matrix: rows are gold labels, columns are predicted labels.
    matrix_labels = CATEGORIES + ["O"]
    short_names = [
        "first", "last", "email", "street", "phone",
        "dob", "company", "other", "none",
    ]
    matrix = confusion_matrix(flat_gold, flat_predictions, labels=matrix_labels)

    print("=== CONFUSION MATRIX (token level; rows = gold, columns = predicted) ===")
    header = f"{'':>9}"
    for name in short_names:
        header += f"{name:>9}"
    print(header)

    for name, row in zip(short_names, matrix):
        line = f"{name:>9}"
        for count in row:
            line += f"{count:>9}"
        print(line)

    # Find the shortest document with at least three PII items.
    # Fall back to the first document if none qualify.
    example_index = 0
    shortest_length = None

    for i in range(len(documents)):
        item_count = 0
        for label in gold_labels[i]:
            if label.startswith("B-"):
                item_count += 1

        document_length = len(documents[i]["tokens"])
        if item_count >= 3:
            if shortest_length is None or document_length < shortest_length:
                example_index = i
                shortest_length = document_length

    print("\n=== EXAMPLE INPUT ===")
    print(test_data[example_index]["text"])
    print("\n=== EXAMPLE OUTPUT ===")
    print(f"{'token':22}{'gold':20}predicted")

    example_tokens = documents[example_index]["tokens"]
    example_gold = gold_labels[example_index]
    example_predictions = predicted_labels[example_index]

    for token, gold, prediction in zip(example_tokens, example_gold, example_predictions):
        mark = ""
        if gold != prediction:
            mark = "   <-- wrong"
        print(f"{token:22}{gold:20}{prediction}{mark}")

    # Compare exact spans to find false positives and false negatives.
    false_positives = []
    false_negatives = []

    for document, gold, predictions in zip(documents, gold_labels, predicted_labels):
        gold_items = set(get_entities(gold))
        predicted_items = set(get_entities(predictions))

        for item in predicted_items - gold_items:
            false_positives.append((item, document["tokens"]))

        for item in gold_items - predicted_items:
            false_negatives.append((item, document["tokens"]))

    random.seed(42)

    print(f"\n=== FALSE POSITIVES ({len(false_positives)} total, showing up to {N_SAMPLES}) ===")
    sample_count = min(N_SAMPLES, len(false_positives))
    for item, tokens in random.sample(false_positives, sample_count):
        show(item, tokens)

    print(f"\n=== FALSE NEGATIVES ({len(false_negatives)} total, showing up to {N_SAMPLES}) ===")
    sample_count = min(N_SAMPLES, len(false_negatives))
    for item, tokens in random.sample(false_negatives, sample_count):
        show(item, tokens)


if __name__ == "__main__":
    main()
