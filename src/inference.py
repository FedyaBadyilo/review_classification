from typing import Any
import torch

from src.utils import preprocess_text, get_rating_label, clean_review, is_english
from models.model import create_model
from src.config import Variables


def get_predictions(
        text: str
) -> tuple[dict[str, Any], None] | tuple[str, True]:
    if not text:
        return "The review can't be empty", True
    if len(text) < 100:
        return 'The review must be longer than 150 characters', True

    cleaned_text = clean_review(text)

    if is_english(cleaned_text):
        return 'The review must be written in English', True

    tokenized_text = preprocess_text(cleaned_text)
    input_ids = tokenized_text['input_ids'].to(Variables.DEVICE)
    attention_mask = tokenized_text['attention_mask'].to(Variables.DEVICE)
    token_type_ids = tokenized_text['token_type_ids'].to(Variables.DEVICE)

    model = create_model()
    model.to(Variables.DEVICE).eval()
    prediction : torch.Tensor = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    ).logits.detach().numpy()[0]

    rating, label = get_rating_label(prediction)

    return {'rating': rating, 'status': Variables.LABELS_LST[label]}, None

