
#  Self-Healing Text Classification Pipeline using LangGraph


##  Project Overview

This project implements a **Self-Healing Text Classification Pipeline** using **LangGraph** and a **fine-tuned transformer model (DistilBERT with LoRA)**.
The system intelligently detects low-confidence predictions and automatically triggers a **human-in-the-loop fallback**, ensuring robust and reliable classification results.

---

##  Objective

> “Build a LangGraph-based classification pipeline that not only performs predictions but also incorporates a self-healing mechanism.”

Key features include:

* Transformer-based text classification (fine-tuned DistilBERT)
* Confidence-based fallback logic using LangGraph DAG
* CLI-driven human-in-the-loop clarification system
* Structured logging for traceability
* Lightweight, efficient LoRA fine-tuning on Google Colab GPU

---
## Uploaded Fine-Tuned Model to Hugging Face
**live model at**
 [Fine-Tune DistilBERT with LoRA —](https://huggingface.co/ranjana1811/fine-tuned-distilbert-lora)

## Architecture Overview

The system is built as a **LangGraph Directed Acyclic Graph (DAG)** with three nodes:

| Node                       | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
|  **InferenceNode**       | Runs text classification using the fine-tuned model.            |
|  **ConfidenceCheckNode** | Evaluates confidence scores and routes control flow.            |
|  **FallbackNode**        | Invokes human clarification when confidence is below threshold. |

---

###  DAG Flow

```
            ┌──────────────────┐
            │  InferenceNode   │
            │  (Prediction)    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ ConfidenceCheck   │
            │ (≥ Threshold ?)   │
            └──────┬──────┬────┘
                   │      │
            (Yes)  │      │ (No)
                   │      ▼
                   │  ┌────────────┐
                   │  │ Fallback   │
                   │  │ (Ask User) │
                   │  └─────┬──────┘
                   ▼        │
                  [END] <───┘
```

---

##  Setup Instructions

###  Prerequisites

* Python ≥ 3.9
* pip ≥ 22.0
* Virtual environment (recommended)

###  Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # macOS/Linux
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Run the Pipeline (CLI)

### 1️⃣ Run the CLI App

```bash
python cli_app.py
```

### 2️⃣ Example Run

```
 Self-Healing Classification Pipeline (LangGraph)
Threshold for Fallback: 70%
Enter 'quit' or 'exit' to stop the application.

[1] Enter text for classification:
> The movie was painfully slow and boring.
[InferenceNode] Predicted label: Negative | Confidence: 68.4%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: Negative (Corrected via user clarification)
```

---

##  Fine-Tuning (Google Colab Notebook)

The model was fine-tuned using LoRA (Low-Rank Adaptation) for efficiency.
Fine-tuning performed on the **IMDB Sentiment Classification Dataset**.

 ## **Colab Notebook Link:**
 [Fine-Tune DistilBERT with LoRA — Google Colab](https://colab.research.google.com/drive/1DwEusZ5c6FQhjEILmypeuATPsjZI6p6l?usp=sharing)

**Key steps:**

1. Load `distilbert-base-uncased`
2. Apply LoRA via PEFT
3. Fine-tune on IMDB sentiment data
4. Save the trained model to Drive or Hugging Face Hub
5. Download and use locally in this project

---

##  Model Integration

The fine-tuned model is loaded using:

```python
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, MODEL_PATH) 

here model path is MODEL_PATH = "ranjana1811/fine-tuned-distilbert-lora"

```

Confidence calibration added:

```python
temperature = 1.5
probabilities = F.softmax(logits / temperature, dim=-1)
```

Confidence smoothing:

```python
confidence_score = round(min(confidence_score, 0.88), 3)
```

---

##  Self-Healing Logic

| Condition          | Action                              |
| ------------------ | ----------------------------------- |
| Confidence ≥ 0.88   | Accept prediction directly          |
| Confidence < 0.88   | Trigger fallback (ask user)         |
| User clarification | Update final label & log correction |

---

##  Logging System

All important events are logged with timestamps in `logs/run_log.txt`.

| Log Type        | Description                           |
| --------------- | ------------------------------------- |
| Inference       | Initial model prediction & confidence |
| ConfidenceCheck | Threshold evaluation                  |
| Fallback        | User clarification stage              |
| Final Decision  | Final accepted or corrected label     |



##  Demo Video

 **Video Link:**
 [Watch Demo ](https://drive.google.com/file/d/YOUR_DEMO_VIDEO_LINK_HERE/view?usp=sharing)



