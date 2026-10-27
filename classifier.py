import ast
import re
from datasets import load_dataset
from huggingface_hub import login
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

def tokenize_with_offsets(text):
    # Words and individual punctuation marks, with their character offsets.
    tokens, offsets = [], []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def create_bio_labels(data):
    results = []

    for row in data:
        text = row["text"]
        spans = ast.literal_eval(row["spans"])

        tokens, offsets = tokenize_with_offsets(text)
        tags = ["O"] * len(tokens)

        for span in spans:
            label = span["label"]

            selected_labels = [
                "first_name",
                "last_name",
                "email",
                "street_address",
                "phone_number",
                "date_of_birth",
                "company_name",
            ]

            if label not in selected_labels:
                label = "other"
            
            first_token = True

            for i in range(len(offsets)):
                token_start, token_end = offsets[i]

                # always sets first token to true so that when token is within an interval range, it always sets first to beginning
                if token_start < span["end"] and token_end > span["start"]:
                    if first_token:
                        tags[i] = "B-" + label
                        first_token = False
                    else:
                        tags[i] = "I-" + label
        results.append({
            "tokens": tokens,
            "bio_labels": tags,
        })

    return results

def train_bert(bio_results):
    set_seed(42)

    categories = [
        "first_name",
        "last_name",
        "email",
        "street_address",
        "phone_number",
        "date_of_birth",
        "company_name",
        "other",
    ]

    label_names = ["O"]

    for category in categories:
        label_names.append("B-" + category)
        label_names.append("I-" + category)

    # label_names = set()

    # for result in bio_results:
    #     for label in result["bio_labels"]:
    #         label_names.add(label)

    # label_names = sorted(label_names)

    # BERT: numeric labels
    label2id = {}
    id2label = {}

    for i in range(len(label_names)):
        label = label_names[i]
        label2id[label] = i
        id2label[i] = label

    # Bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def prepare_doc(doc):
        encoded = tokenizer(
            doc["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        labels = []
        previous_word = None

        for word in encoded.word_ids():
            if word is None:
                labels.append(-100)
            elif word == previous_word:
                labels.append(-100)
            else: 
                label = doc["bio_labels"][word]
                labels.append(label2id[label])

            previous_word = word

        encoded["labels"] = labels

        return encoded

    training_data = Dataset.from_list(bio_results)

    # prepare docs for BERT
    training_data = training_data.map(
        prepare_doc,
        remove_columns=training_data.column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "google-bert/bert-base-cased",
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    settings = TrainingArguments(
        output_dir="./bert_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=settings,
        train_dataset=training_data,
        data_collator=collator,
    )

    print("Training Device: ", settings.device)

    trainer.train()

    trainer.save_model("./bert_pii_model")
    tokenizer.save_pretrained("./bert_pii_model")

# ____________________________________________________________________________

def main():
    # login() # login to huggingface
    dataset = load_dataset("nvidia/Nemotron-PII") # receive dataset Nemotron-PII
    training_data = dataset["train"].select_columns(["text", "spans"]) # receveies training data with just text and spans columns

    print("Training documents:", len(training_data), flush=True)

    bio_results = create_bio_labels(training_data)

    # for result in bio_labels[:2]:
    #     for i in range(len(result["tokens"])):
    #         token = result["tokens"][i]
    #         label = result["bio_labels"][i]
    #         print(f"{token:20} {label}")

    train_bert(bio_results)

if __name__ == "__main__":
    main()