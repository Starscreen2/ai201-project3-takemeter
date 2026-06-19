"""Gradio demo for TakeMeter fine-tuned classifier."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("strategy_tip", "hot_take", "reaction")
MODEL_DIR = Path(__file__).resolve().parents[1] / "finetuned_model"


def load_model():
    model_path = os.environ.get("TAKEMETER_MODEL_DIR", str(MODEL_DIR))
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Export from Colab or run scripts/train_eval.py."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def classify(text: str):
    if not text.strip():
        return "Enter some text first.", ""

    tokenizer, model = load_model()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    breakdown = "\n".join(
        f"- {model.config.id2label[index]}: {float(probs[index]):.1%}"
        for index in range(len(probs))
    )
    return f"**{label}** ({confidence:.1%})", breakdown


def main() -> None:
    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=5, label="r/leagueoflegends post or comment"),
        outputs=[
            gr.Markdown(label="Prediction"),
            gr.Markdown(label="All label probabilities"),
        ],
        title="TakeMeter",
        description="Classify LoL discourse as strategy_tip, hot_take, or reaction.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
