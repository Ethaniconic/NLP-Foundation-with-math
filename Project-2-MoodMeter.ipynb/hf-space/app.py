import torch
import torch.nn as nn
import re
import json
import os
import gradio as gr
from pathlib import Path

# ─────────────────────────────────────────────
#  Model definition  (mirrors src/model.py)
# ─────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded   = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        out        = lstm_out[:, -1, :]
        out        = self.dropout(out)
        out        = self.fc(out)
        return self.sigmoid(out)


# ─────────────────────────────────────────────
#  Load vocab + model  (once at startup)
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
VOCAB_PATH  = BASE_DIR / "vocab.json"
MODEL_PATH  = BASE_DIR / "best_model.pth"
MAX_LENGTH  = 100

print("Loading vocab ...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab: dict = json.load(f)

print(f"  vocab size: {len(vocab):,}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device} ...")

model = SentimentLSTM(vocab_size=len(vocab))
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

val_acc = checkpoint.get("val_acc", None)
print(f"  Model loaded  -  val acc: {val_acc:.4f}" if val_acc else "  Model loaded")


# ─────────────────────────────────────────────
#  Inference helper
# ─────────────────────────────────────────────
def preprocess(text: str) -> torch.Tensor:
    text    = text.lower()
    text    = re.sub(r"[^a-z\s]", "", text)
    words   = text.split()
    indices = [vocab.get(w, vocab["<UNK>"]) for w in words]

    if len(indices) < MAX_LENGTH:
        indices = indices + [vocab["<PAD>"]] * (MAX_LENGTH - len(indices))
    else:
        indices = indices[:MAX_LENGTH]

    return torch.tensor([indices], dtype=torch.long)


def predict_sentiment(review: str):
    """Core prediction function used by Gradio."""
    review = review.strip()
    if not review:
        return (
            "*Please enter some text.*",
            {"Positive": 0.0, "Negative": 0.0},
            ""
        )

    tensor = preprocess(review).to(device)

    with torch.no_grad():
        raw = model(tensor).item()          # 0 = negative, 1 = positive

    if raw >= 0.5:
        sentiment  = "Positive"
        emoji      = "😊"
        confidence = raw
    else:
        sentiment  = "Negative"
        emoji      = "😞"
        confidence = 1.0 - raw

    label_probs = {
        "Positive 😊": raw,
        "Negative 😞": 1.0 - raw,
    }

    verdict = f"## {emoji} **{sentiment}**\n\n**Confidence:** `{confidence:.1%}`"

    return verdict, label_probs, f"{confidence:.1%}"


# ─────────────────────────────────────────────
#  Example reviews
# ─────────────────────────────────────────────
EXAMPLES = [
    ["This movie was absolutely breathtaking! The performances were stellar and the story kept me hooked till the very end."],
    ["Terrible film. The plot made no sense, the acting was wooden, and the CGI was embarrassing. Complete waste of time."],
    ["A masterpiece of modern cinema. One of the best films I have seen in years. Highly recommended!"],
    ["I fell asleep halfway through. Incredibly boring with no redeeming qualities whatsoever."],
    ["Surprisingly decent despite the mixed reviews. Not perfect, but enjoyable enough for a Saturday night."],
    ["One of the worst movies ever made. Predictable, cliched, and utterly forgettable."],
]

# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.hero-title {
    background: linear-gradient(90deg, #a78bfa, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    text-align: center;
    line-height: 1.1;
    margin: 0 0 6px 0;
}
.hero-sub {
    text-align: center;
    color: #c4b5fd;
    font-size: 1.05rem;
    margin-bottom: 24px;
}
textarea {
    border-radius: 12px !important;
    font-size: 1rem !important;
}
#analyse-btn {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    color: white !important;
}
#analyse-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.55) !important;
}
#verdict-md {
    font-size: 1.5rem;
    padding: 16px;
    text-align: center;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.82rem;
    margin-top: 8px;
}
"""

with gr.Blocks(title="MoodMeter - Sentiment Analyzer") as demo:

    gr.HTML("""
    <div class="card">
        <p class="hero-title">🎬 MoodMeter</p>
        <p class="hero-sub">
            AI-powered Movie Review Sentiment Analyzer &nbsp;&middot;&nbsp;
            LSTM trained on 25,000 IMDB reviews
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            review_box = gr.Textbox(
                label="🎞️ Movie Review",
                placeholder="Type or paste a movie review here...",
                lines=6,
                max_lines=12,
                elem_id="review-box",
            )

            analyse_btn = gr.Button(
                "✨  Analyse Sentiment",
                variant="primary",
                elem_id="analyse-btn",
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=review_box,
                label="💡 Try an example",
            )

        with gr.Column(scale=2):
            verdict_md = gr.Markdown(
                value="*Results will appear here...*",
                elem_id="verdict-md",
            )

            label_output = gr.Label(
                label="📊 Sentiment Probabilities",
                num_top_classes=2,
            )

            confidence_txt = gr.Textbox(
                label="🎯 Confidence Score",
                interactive=False,
            )

    info_text = f"""
**Architecture:** Custom single-layer LSTM with learned word embeddings

| Hyperparameter | Value |
|---|---|
| Embedding dim | 100 |
| Hidden dim | 128 |
| Vocabulary size | {len(vocab):,} |
| Max sequence length | {MAX_LENGTH} |
| Dropout | 0.5 |
| Trained on | IMDB (25 k reviews) |
| Validation accuracy | {f"{val_acc:.2%}" if val_acc else "N/A"} |

The model tokenises input text, maps tokens to dense embeddings, passes them through an LSTM,
and feeds the final hidden state through a sigmoid classifier to produce a sentiment probability.
"""

    with gr.Accordion("ℹ️  About this model", open=False):
        gr.Markdown(info_text)

    gr.HTML('<p class="footer">Built with PyTorch &amp; Gradio &nbsp;&middot;&nbsp; MoodMeter v1.0</p>')

    analyse_btn.click(
        fn=predict_sentiment,
        inputs=[review_box],
        outputs=[verdict_md, label_output, confidence_txt],
        api_name="predict",
    )

    review_box.submit(
        fn=predict_sentiment,
        inputs=[review_box],
        outputs=[verdict_md, label_output, confidence_txt],
    )


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
