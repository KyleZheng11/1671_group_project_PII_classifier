from transformers import pipeline

pii = pipeline("token-classification", model="./bert_pii_model_full", aggregation_strategy="first", stride=128)

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
        print(f"{item['start']:>6} {item['end']:>6}  {text[item['start']:item['end']]:40} "
              f"{item['entity_group']:15} {item['score']:.2f}")