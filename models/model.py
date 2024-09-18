import transformers
import pickle
import os

from src.config import Variables


def create_model() -> transformers.BertForSequenceClassification:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if 'models' in str(current_dir):
        file_name = os.path.join(current_dir, 'bert_model.pkl')
    else:
        file_name = os.path.join(current_dir, 'models', 'bert_model.pkl')
    with open(file_name, 'rb') as model_file:
        model = pickle.load(model_file)
    model.to(Variables.DEVICE)
    return model


def create_tokenizer() -> transformers.BartTokenizer:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    if 'models' in str(current_dir):
        file_name = os.path.join(current_dir, 'tokenizer.pkl')
    else:
        file_name = os.path.join(current_dir, 'models', 'tokenizer.pkl')
    with open(file_name, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer
