import os

from transformers import pipeline

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
REDACTION = "[REDACTED]"

pii = pipeline(
    "token-classification",
    model="./bert_pii_model_full",
    aggregation_strategy="first",
    stride=128,
)


def get_text():
    """Accept raw text, or a path to a .txt file (so multi-line documents work)."""
    raw = input("\nEnter text or a path to a text file (or 'quit'): ").strip()
    if raw.lower() == "quit":
        return None
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as f:
            return f.read()
    return raw


def choose_categories():
    """Show the numbered menu and return the set of chosen labels."""
    print("\nWhich PII should be removed?")
    for i, name in enumerate(CATEGORIES, start=1):
        print(f"  {i}: {name.replace('_', ' ')}")

    while True:
        choice = input("Enter numbers separated by commas (e.g. 1,3,5), or 'all': ").strip()
        if choice.lower() == "all":
            return set(CATEGORIES)
        try:
            nums = {int(n) for n in choice.split(",") if n.strip()}
        except ValueError:
            print("Please enter numbers only, like 1,3,5.")
            continue
        if not nums or any(n < 1 or n > len(CATEGORIES) for n in nums):
            print(f"Numbers must be between 1 and {len(CATEGORIES)}.")
            continue
        return {CATEGORIES[n - 1] for n in nums}


def redact(text, labels):
    """Replace every detected span whose label is in `labels` with [REDACTED]."""
    spans = sorted(
        (r["start"], r["end"]) for r in pii(text) if r["entity_group"] in labels
    )

    # Merge overlapping spans so nothing gets replaced twice
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Replace from the end backwards so earlier offsets stay valid
    for start, end in reversed(merged):
        text = text[:start] + REDACTION + text[end:]
    return text, len(merged)


while True:
    text = get_text()
    if text is None:
        break
    if not text:
        continue

    labels = choose_categories()
    redacted, count = redact(text, labels)

    print(f"\nRemoving: {', '.join(sorted(labels))}  ({count} span(s) redacted)\n")
    print(redacted)