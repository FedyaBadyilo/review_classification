from fastapi import FastAPI, Request, Form
from pydantic import BaseModel

from src.inference import get_predictions
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class Review(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, review_text: str = Form(...)):
    predictions, warning = get_predictions(review_text)

    if warning:
        return templates.TemplateResponse("form.html", {"request": request, "warning": predictions})

    return templates.TemplateResponse("form.html", {"request": request, "predictions": predictions})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


