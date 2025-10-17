# utils/config.py

import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent # /self_healing_classifier
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "run_log.txt"
MODEL_PATH = BASE_DIR / "fine_tuned_distilbert_lora"
# If using a model from Hugging Face Hub, we  
 # can use this directly MODEL_PATH = "https://huggingface.co/ranjana1811/fine-tuned-distilbert-lora"

# Create logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)

# --- Model & Workflow Settings ---
# Confidence threshold for triggering the FallbackNode (Requirement [cite: 15])
# The prediction confidence must be >= this value to be ACCEPTED.
CONFIDENCE_THRESHOLD = 0.88 

# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
LABEL_MAP = {0: "Negative", 1: "Positive"} # Based on IMDB dataset

# --- LangGraph Node Names ---
INFERENCE_NODE = "InferenceNode"
CONFIDENCE_CHECK_NODE = "ConfidenceCheckNode"
FALLBACK_NODE = "FallbackNode"

# --- Output Tags ---
CORRECTED_TAG = "(Corrected via user clarification)"