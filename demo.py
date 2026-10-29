import csv
import os

from transformers import pipeline

CSV_PATH = "output/pii_results.csv"
FIELDS = ["start", "end", "text", "label", "score"]

pii = pipeline(
    "token-classification",
    model="./bert_pii_model_full",
    aggregation_strategy="first",
    stride=128,
)

# Append to the CSV across runs; write the header only if the file is new/empty
write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0

with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    while True:
        text = input("\nEnter text (or 'quit'): ")
        if text.lower() == "quit":
            break

        results = pii(text)

        if not results:
            print("No PII found.")
            continue

        print(f"{'start':>6} {'end':>6}  {'text':40} {'label':15} score")
        for item in results:
            span = text[item["start"]:item["end"]]
            print(f"{item['start']:>6} {item['end']:>6}  {span:40} "
                  f"{item['entity_group']:15} {item['score']:.2f}")

            writer.writerow({
                "start": item["start"],
                "end": item["end"],
                "text": span,
                "label": item["entity_group"],
                "score": round(float(item["score"]), 4),
            })

        f.flush()  # save rows now, so nothing is lost if the script crashes
        print(f"Saved {len(results)} row(s) to {CSV_PATH}")