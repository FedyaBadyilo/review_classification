from transformers import BatchEncoding
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

from models.model import create_tokenizer
from src.config import Variables


def preprocess_text(text: str) -> BatchEncoding:
    tokenizer = create_tokenizer()
    text_tokenized = tokenizer(text, **Variables.TOKENIZER_PARAMS, )
    return text_tokenized


def get_rating_label(arr: np.array) -> tuple:
    rating = np.argmax(arr, axis=0)
    if rating <= 3:
        rating += 1
        label = 0
    else:
        rating += 3
        label = 1

    return rating, label


def clean_review(review):
    review = BeautifulSoup(review, "lxml").get_text()
    review = re.sub("@[A-Za-z0-9_]+", "", review)

    review = re.sub("https?://[^ ]+", "", review)
    review = re.sub("www.[^ ]+", "", review)

    try:
        review = review.decode("utf-8-sig")
    except:
        pass

    review.replace(u"\ufffd", "?")
    review = re.sub("[^a-zA-Z]"," ", review)
    words = WordPunctTokenizer().tokenize(review.lower())

    cleaned_review = (" ".join(words)).strip()

    return cleaned_review


def is_english(text):
    try:
        if len(set(text) & set(Variables.ENGLISH_LETTERS)) / len(set(text)) < 0.8:
            return False
    except ZeroDivisionError:
        return False
    return True


if __name__ == '__main__':
    print(is_english(clean_review('NIGGER NIGGERS')))



