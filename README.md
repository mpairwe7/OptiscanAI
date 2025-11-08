---
title: Retinal Screening
emoji: 👁
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.1"
app_file: src/streamlit_app.py
pinned: false
license: mit
short_description: ML for retinal screening in Ugandan health care
---

# Retinal Screening ML Application

This is a machine learning application for retinal screening using computer vision models to assist in healthcare diagnostics.

## Features

- Deep learning models for retinal image analysis
- Streamlit web interface
- Docker deployment ready
- Model explanations and visualizations

## Models

- Visual Language GNN
- GraphCLIP Rank models
- Mobile-optimized models

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `streamlit run streamlit_app.py`
3. Or use Docker: `docker-compose up`

## Deployment

Deployed on Hugging Face Spaces with Git LFS for large model files.