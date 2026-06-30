---
title: MoodMeter - Sentiment Analyzer
emoji: 🎬
colorFrom: violet
colorTo: pink
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
license: mit
short_description: LSTM-based movie review sentiment analyzer trained on IMDB
---

# 🎬 MoodMeter — Movie Review Sentiment Analyzer

A sentiment analysis model trained from scratch using a custom LSTM architecture on the IMDB dataset.

## Model Architecture

| Component | Detail |
|---|---|
| Architecture | Single-layer LSTM |
| Embedding dim | 100 |
| Hidden dim | 128 |
| Vocabulary | 10,000 tokens |
| Max sequence length | 100 tokens |
| Dropout | 0.5 |
| Dataset | IMDB (25,000 reviews) |
| Framework | PyTorch |

## How it works

1. Input text is lowercased and stripped of non-alphabetic characters
2. Tokens are mapped to integer indices via a vocabulary built from the training corpus
3. Embeddings are passed through an LSTM encoder
4. The final hidden state is fed through a sigmoid classifier
5. Output ≥ 0.5 → Positive, < 0.5 → Negative

## Usage

Type or paste a movie review into the text box and click **Analyse Sentiment**.
