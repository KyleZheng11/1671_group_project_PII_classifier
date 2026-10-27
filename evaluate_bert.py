from transformers import AutoTokenizer, AutoModelForTokenClassification

#-----
import random
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report as token_report, confusion_matrix
from seqeval.metrics import classification_report as span_report
from seqeval.metrics.sequence_labeling import get_entities
from classifier import create_bio_labels


MODEL_PATH = "./bert_pii_model_full"
N_DEV = 100      # change to 1000 once it works
N_SAMPLES = 10   # how many false positives / false negatives to print
CATEGORIES = ["first_name", "last_name", "email", "street_address", "phone_number", "date_of_birth", "company_name", "other"]


def predict(tokens, tokenizer, model):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
    with torch.no_grad():
        label_ids = model(**enc).logits[0].argmax(-1).tolist()

    preds = ["O"] * len(tokens)
    previous = None
    for pos, word in enumerate(enc.word_ids()):
        if word is not None and word != previous:
            preds[word] = model.config.id2label[label_ids[pos]]
        previous = word
    return preds


def show(item, tokens):
    label, start, end = item
    span = " ".join(tokens[start:end + 1])
    context = " ".join(tokens[max(0, start - 6):end + 7])
    print(f"  [{label}] '{span}'\n      context: ...{context}...")




def main():

    tokenizer = AutoTokenizer.from_pretrained("./bert_pii_model_full")
    model = AutoModelForTokenClassification.from_pretrained("./bert_pii_model_full")
    model.eval()

    print("Model and tokenizer")
    print("Number of labels:", model.config.num_labels)
    print("Labels:", model.config.id2label)

       # Test documents
    data = load_dataset("nvidia/Nemotron-PII")
    dev = data["test"].select_columns(["text", "spans"]).shuffle(seed=42).select(range(N_DEV))
    docs = create_bio_labels(dev)
    gold = [doc["bio_labels"] for doc in docs]

    n_gold_items = sum(tag.startswith("B-") for seq in gold for tag in seq)
    print(f"Documents tested: {len(docs)}   Gold PII items: {n_gold_items}")

    # BERT predictions
    pred = []
    for i, doc in enumerate(docs):
        pred.append(predict(doc["tokens"], tokenizer, model))
        if (i + 1) % 25 == 0:
            print(f"Predicted {i + 1}/{len(docs)}", flush=True)

    # 1. Metrics per category
    flat_gold = [t[2:] if t != "O" else "O" for seq in gold for t in seq]
    flat_pred = [t[2:] if t != "O" else "O" for seq in pred for t in seq]
    print("\n=== TOKEN LEVEL (per category) ===")
    print(token_report(flat_gold, flat_pred, labels=CATEGORIES, digits=4, zero_division=0))
    print("=== SPAN LEVEL (exact match, per category) ===")
    print(span_report(gold, pred, digits=4, zero_division=0))

    # 2. Confusion matrix (rows = correct category, columns = BERT's prediction)
    labels = CATEGORIES + ["O"]
    short = ["first", "last", "email", "street", "phone", "dob", "company", "other", "none"]
    cm = confusion_matrix(flat_gold, flat_pred, labels=labels)
    print("=== CONFUSION MATRIX (token level; rows = gold, columns = predicted) ===")
    print(f"{'':>9}" + "".join(f"{s:>9}" for s in short))
    for name, row in zip(short, cm):
        print(f"{name:>9}" + "".join(f"{n:>9}" for n in row))

    # 3. Example input and output: shortest document with at least 3 PII items
    candidates = [i for i, g in enumerate(gold) if sum(t.startswith("B-") for t in g) >= 3]
    ex = min(candidates, key=lambda i: len(docs[i]["tokens"])) if candidates else 0
    print("\n=== EXAMPLE INPUT ===")
    print(dev[ex]["text"])
    print("\n=== EXAMPLE OUTPUT ===")
    print(f"{'token':22}{'gold':20}predicted")
    for tok, g, p in zip(docs[ex]["tokens"], gold[ex], pred[ex]):
        mark = "" if g == p else "   <-- wrong"
        print(f"{tok:22}{g:20}{p}{mark}")

    # 4. False positives and false negatives
    false_pos, false_neg = [], []
    for doc, g, p in zip(docs, gold, pred):
        gold_items, pred_items = set(get_entities(g)), set(get_entities(p))
        false_pos += [(item, doc["tokens"]) for item in pred_items - gold_items]
        false_neg += [(item, doc["tokens"]) for item in gold_items - pred_items]

    random.seed(42)
    print(f"\n=== FALSE POSITIVES ({len(false_pos)} total, showing up to {N_SAMPLES}) ===")
    for item, tokens in random.sample(false_pos, min(N_SAMPLES, len(false_pos))):
        show(item, tokens)
    print(f"\n=== FALSE NEGATIVES ({len(false_neg)} total, showing up to {N_SAMPLES}) ===")
    for item, tokens in random.sample(false_neg, min(N_SAMPLES, len(false_neg))):
        show(item, tokens)




if __name__ == "__main__":
    main()