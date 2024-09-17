import torch


class Variables:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BERT_PATH = 'bhadresh-savani/bert-base-uncased-emotion'
    NUM_LABELS = 8
    TOKENIZER_PARAMS = {'max_length': 256,
                        'padding': "max_length",
                        'truncation': 'only_first',
                        'return_tensors': 'pt',
                        }

    LABELS_LST = ['Negative', 'Positive']
    ENGLISH_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'



