"""
classifier.py -- fine-tune bert-base-cased to find PII spans in nvidia/Nemotron-PII.

What this file does, in order:
  1. Loads the dataset (each row has `text` and `spans`; spans are CHARACTER ranges with a label).
  2. Turns each text into regex "words" and gives every word a BIO tag (preprocessing).
  3. Runs BERT's own subword tokenizer, splitting long documents into overlapping windows,
     and lines the word-level tags up with the subword tokens.
  4. Trains BERT with the Hugging Face Trainer, keeping the best epoch on a validation set.
  5. Saves the model to SAVE_PATH.

Evaluation lives in evaluate_bert.py, which imports the settings and helper functions from here.

Install: pip install torch transformers datasets seqeval accelerate
Run:     python classifier.py
"""

import ast
import os
import re
from bisect import bisect_left

import torch
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =============================================================================
# SETTINGS (evaluate_bert.py imports these, so both files always agree)
# =============================================================================

MODEL_NAME = "google-bert/bert-base-cased"  # "cased" keeps capital letters, which help for names/companies
SAVE_PATH = "./bert_pii_model"              # where the trained model is saved

MAX_LEN = 512   # BERT can read at most 512 tokens at once (this includes [CLS] and [SEP])
STRIDE = 128    # when a document is longer than MAX_LEN, consecutive windows share this many tokens

VAL_FRACTION = 0.05  # share of the train split held out to choose the best epoch
# Quick test run:  SUBSET=2000 python classifier.py     (unset / 0 = use the whole dataset)
SUBSET = int(os.environ.get("SUBSET", 0)) or None

# We keep 7 PII types as their own classes; every other type in the dataset becomes "other".
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
SELECTED_LABELS = CATEGORIES[:-1]  # everything not in this list is mapped to "other"

# BIO tagging: "B-x" = first word of a span of type x, "I-x" = later words of that span,
# "O" = not PII. Each category therefore needs two labels (B- and I-), plus one shared "O".
LABEL_NAMES = ["O"]
for category in CATEGORIES:
    LABEL_NAMES.append("B-" + category)
    LABEL_NAMES.append("I-" + category)

# The model works with integers, so we need a dictionary in each direction.
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# =============================================================================
# STEP 1: PREPROCESSING -- character spans  ->  one BIO tag per word
# =============================================================================

def tokenize_with_offsets(text):
    """Split text into 'words' and remember where each word sits in the original text.

    A word is a run of letters/digits/underscore, or a single punctuation mark.
    We need the (start, end) character positions because the dataset's spans are
    given in characters, so they are the only way to tell which words are PII.
    """
    tokens, offsets = [], []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def resolve_overlaps(spans):
    """Make sure no character belongs to more than one span.

    Some rows contain spans that overlap (e.g. a full-name span containing a first-name span).
    If we tagged them one after another, the later span would overwrite the earlier one and
    could leave broken tag sequences (an "I-x" with no "B-x" before it).
    Rule: sort by start position (longer span first on ties) and skip any span that
    starts before the previously kept span has ended. This keeps the OUTER span.
    """
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept, last_end = [], -1
    for span in spans:
        if span["start"] >= last_end:
            kept.append(span)
            last_end = span["end"]
    return kept


def create_bio_labels(example):
    """Turn one dataset row into word-level tokens and BIO tags.

    Takes a row with `text` and `spans`; returns {"tokens": [...], "bio_labels": [...]}.
    Designed for dataset.map(create_bio_labels), which adds the two new columns.
    """
    text = example["text"]

    # `spans` is stored as a STRING that looks like a Python list of dicts, so parse it first.
    spans = resolve_overlaps(ast.literal_eval(example["spans"]))

    tokens, offsets = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)                    # start with everything as "not PII"
    token_ends = [end for _, end in offsets]      # sorted, because words never overlap

    for span in spans:
        label = span["label"] if span["label"] in SELECTED_LABELS else "other"
        first_word = True

        # Jump straight to the first word whose end is after the span start
        # (bisect on the sorted ends instead of scanning every word for every span).
        i = bisect_left(token_ends, span["start"] + 1)

        # Walk forward over every word that starts before the span ends.
        while i < len(offsets) and offsets[i][0] < span["end"]:
            tags[i] = ("B-" if first_word else "I-") + label  # first word gets B-, the rest I-
            first_word = False
            i += 1

    return {"tokens": tokens, "bio_labels": tags}


# =============================================================================
# STEP 2: BERT TOKENIZATION -- words  ->  subword tokens, with aligned labels
# =============================================================================

def tokenize_and_align_labels(examples, tokenizer):
    """Give BERT the words, and give every subword token the right label id.

    BERT splits words further into pieces ("Jasonville" -> "Jason", "##ville") and adds
    [CLS] / [SEP], so there are more BERT tokens than words and we must re-line-up the labels.
    Runs on a BATCH of rows (dataset.map(..., batched=True)), so `examples` holds lists.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,         # input is already split into words; keep track of which word each piece came from
        truncation=True,
        max_length=MAX_LEN,
        stride=STRIDE,                    # windows of a long document overlap by this many tokens
        return_overflowing_tokens=True,   # a long document becomes SEVERAL windows instead of being cut off
    )

    # For each window: the index of the document in this batch that it came from.
    window_to_doc = tokenized.pop("overflow_to_sample_mapping")

    all_labels = []
    for window_index, doc_index in enumerate(window_to_doc):
        bio_labels = examples["bio_labels"][doc_index]          # word-level tags of THAT document
        word_ids = tokenized.word_ids(batch_index=window_index)  # for each BERT token: which word it belongs to

        labels = []
        previous_word = None
        for word in word_ids:
            if word is None:
                labels.append(-100)                              # [CLS]/[SEP]: -100 means "ignore in the loss"
            elif word != previous_word:
                labels.append(LABEL2ID[bio_labels[word]])        # first subword of a word gets the word's label
            else:
                labels.append(-100)                              # later subwords of the same word are ignored
            previous_word = word
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


# =============================================================================
# STEP 3: METRICS -- span-level precision / recall / F1 (seqeval)
# =============================================================================

def preprocess_logits_for_metrics(logits, labels):
    # The model outputs 17 scores per token. We only need the winning label id,
    # and keeping only that saves a lot of memory during evaluation.
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Called by the Trainer after each epoch on the validation set.

    Accuracy would be misleading (most tokens are "O"), so we use seqeval, which counts a
    predicted span as correct only if its type AND exact boundaries match the gold span.
    """
    predictions, labels = eval_pred  # predictions are already label ids (see function above)

    true_predictions, true_labels = [], []
    for pred_row, label_row in zip(predictions, labels):
        # Skip positions with label -100 (special tokens and non-first subwords).
        true_predictions.append([ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100])
        true_labels.append([ID2LABEL[l] for p, l in zip(pred_row, label_row) if l != -100])

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


# =============================================================================
# STEP 4: TRAINING
# =============================================================================

def train_bert(train_full):
    set_seed(42)

    # Hold out a small validation set from train. It is used to pick the best epoch,
    # so the dataset's real test split stays untouched until evaluate_bert.py.
    split = train_full.train_test_split(test_size=VAL_FRACTION, seed=42)
    train_data, val_data = split["train"], split["test"]

    # Preprocessing step 1: add `tokens` and `bio_labels` columns (word level).
    train_data = train_data.map(create_bio_labels)
    val_data = val_data.map(create_bio_labels)

    # Preprocessing step 2: BERT subword tokenization + label alignment (window level).
    # batched=True processes many rows at once; we drop the old columns because one
    # document can become several windows, so the row counts no longer match.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = train_data.map(
        tokenize_and_align_labels, batched=True,
        fn_kwargs={"tokenizer": tokenizer}, remove_columns=train_data.column_names,
    )
    tokenized_val = val_data.map(
        tokenize_and_align_labels, batched=True,
        fn_kwargs={"tokenizer": tokenizer}, remove_columns=val_data.column_names,
    )
    print(f"train: {len(train_data)} documents -> {len(tokenized_train)} windows")
    print(f"val:   {len(val_data)} documents -> {len(tokenized_val)} windows")

    # BERT body + a new linear layer that maps each token's embedding to our 17 labels.
    # (Warnings about UNEXPECTED / MISSING weights when loading are normal: the new
    # classification layer starts random and is what training teaches.)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Pads every batch to its longest window, and pads the labels with -100 so padding is ignored.
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Mixed precision makes GPU training much faster. bf16 is used when the GPU supports it
    # (A100/H100/L40...), otherwise fp16. On a CPU both stay off.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    settings = TrainingArguments(
        output_dir="./bert_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,   # lower this if you run out of memory
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",            # evaluate on the validation set after every epoch...
        save_strategy="epoch",            # ...save a checkpoint after every epoch...
        save_total_limit=2,
        load_best_model_at_end=True,      # ...and reload the checkpoint with the best F1 when finished
        metric_for_best_model="f1",
        logging_steps=100,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        dataloader_pin_memory=use_cuda,   # faster CPU->GPU copies when a GPU is present
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=settings,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Training device:", settings.device)
    trainer.train()

    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Saved model to", SAVE_PATH)


# =============================================================================

def main():
    dataset = load_dataset("nvidia/Nemotron-PII")
    train_full = dataset["train"].select_columns(["text", "spans"])  # only the two columns we need

    if SUBSET:
        train_full = train_full.select(range(SUBSET))

    print("Training documents:", len(train_full), flush=True)
    train_bert(train_full)


if __name__ == "__main__":
    main()
