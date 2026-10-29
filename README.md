### Files
* bert_pii_model_full: bert model trained on full training set, the other bert files were trained on a small set to test
* classifier.py: contains code to tokenize and create bio labels for the training set. These are used int he train_bert function in order to train our BERT model.
* evaluate_bert.py: evaluates the BERT model and returns precision, recall, f1-score, and support for the bio labels. Also provides confusion matrix, example inputs, and example outputs.


### Generative AI Use
- As stated in the class website, we used LLMs to help generate portions of our code. We utilized ChatGPT and Claude to help generate portions of the code involving the BERT training and BERT evaluation. 