# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:24:21 2024

@author: Akash
"""

import streamlit as st
from transformers import pipeline
import torch

# Load the sentiment analysis pipeline (outside the request handler for efficiency)
try:
    sentiment_model = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.title("Sentiment Analysis App")

text_input = st.text_area("Enter your text here:", height=200)

if st.button("Analyze"):
    if not text_input:
        st.warning("Please enter some text.")
    else:
        try:
            results = sentiment_model(text_input)
            st.write("Results:")
            for result in results:
                st.write(f"Label: {result['label']}, Score: {result['score']:.4f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")