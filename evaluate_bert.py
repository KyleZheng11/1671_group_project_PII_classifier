from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():

    tokenizer = AutoTokenizer.from_pretrained("./bert_pii_model")

    model = AutoModelForTokenClassification.from_pretrained("./bert_pii_model")

    model.eval()

    print("Model and tokenizer")
    print("Number of labels:", model.config.num_labels)
    print("Labels:", model.config.id2label)


if __name__ == "__main__":
    main()