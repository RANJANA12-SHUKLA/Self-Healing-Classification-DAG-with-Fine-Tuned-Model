# dag_pipeline.py

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from utils.config import (
    MODEL_PATH, MODEL_NAME, NUM_LABELS, LABEL_MAP,
    CONFIDENCE_THRESHOLD, INFERENCE_NODE, CONFIDENCE_CHECK_NODE,
    FALLBACK_NODE, CORRECTED_TAG
)
from utils.logger import log_event, logging

# ----------------------------------------------------------------------
# 1. Define the Graph State (The shared data structure)
# ----------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Represents the state of our workflow.
    """
    text_input: str                 # The original text provided by the user
    prediction: str                 # The model's initial predicted label
    confidence_score: float         # The model's confidence in the prediction (0.0 to 1.0)
    final_label: str                # The final classification label (corrected or original)
    user_clarification: str         # Stores user input during fallback
    step_history: Annotated[List[str], lambda x, y: x + y] # Tracks the path taken



    # Load Model and Tokenizer (runs once)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Load LoRA adapters onto the base model
# This uses the files you saved in your 'fine_tuned_distilbert_lora' folder
try:
    model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
    model.eval() # Set model to evaluation mode
    print(f"Successfully loaded LoRA model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading LoRA model: {e}")
    print("Ensure you have unzipped your model files into the 'fine_tuned_distilbert_lora' directory.")
    # Exit or raise error if model loading fails
    raise


# ----------------------------------------------------------------------
# A. Inference Node   (FINAL POLISHED VERSION)
# ----------------------------------------------------------------------
def run_inference(state: GraphState) -> GraphState:
    """Runs classification using the trained model."""

    text = state["text_input"]
    log_event(INFERENCE_NODE, f"Running inference for text: '{text[:50]}...'")

    # 1. Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    # 2. Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    #probabilities = F.softmax(logits, dim=-1)

    # ----------------------------------------------------
    # ADDED: Confidence Smoothing (Temperature Scaling)
    # ----------------------------------------------------
    T = 1.2 # Set temperature > 1.0 to smooth the probabilities
    probabilities = F.softmax(logits / T, dim=-1) # Apply smoothing

    # Get highest probability and label
    confidence, predicted_index = torch.max(probabilities, dim=1)
    confidence_score = confidence.item()
    predicted_label = LABEL_MAP[predicted_index.item()]

    #  Optional smoothing for more realistic confidence display
    confidence_score = round(min(confidence_score, 0.98), 3)

    log_event(
        INFERENCE_NODE,
        f"Predicted label: {predicted_label} | Confidence: {confidence_score * 100:.1f}%",
    )

    return {
        "prediction": predicted_label,
        "confidence_score": confidence_score,
        "final_label": predicted_label,  #  store it once for CLI display
        "step_history": [INFERENCE_NODE],
    }

#-----------------------------------------------------------------------------
# B. Confidence Check Node
#-----------------------------------------------------------------------------
def check_confidence(state: GraphState) -> GraphState:
    """Evaluates confidence and adds decision to state for routing."""
    
    confidence = state["confidence_score"]
    decision = "accept" if confidence >= CONFIDENCE_THRESHOLD else "fallback"

    if decision == "accept":
        log_event(CONFIDENCE_CHECK_NODE, 
                  f"Confidence {confidence:.2f} >= Threshold {CONFIDENCE_THRESHOLD:.2f}. Accepting prediction.")
    else:
        log_event(CONFIDENCE_CHECK_NODE, 
                  f"Confidence {confidence:.2f} < Threshold {CONFIDENCE_THRESHOLD:.2f}. Triggering fallback...")

    #  Return a dict that LangGraph can merge into state
    return {
        "decision": decision,
        "step_history": [CONFIDENCE_CHECK_NODE],
    }

#----------------------------------------------------------------------------
# C. Fallback Node
#----------------------------------------------------------------------------
def trigger_fallback(state: GraphState) -> GraphState:
    """
    Avoids incorrect classification by preparing the state for user clarification.
    Note: The actual question/answer happens in the CLI interface.
    """
    log_event(FALLBACK_NODE, 
              f"Preparing for user clarification. Initial prediction was '{state['prediction']}'.")
              
    # The state will be returned to the CLI, which will prompt the user.
    return {
        "user_clarification": "PENDING", # Flag to indicate user input is needed
        "step_history": [FALLBACK_NODE]
    }


# ----------------------------------------------------------------------
# 4. Build the LangGraph Workflow
# ----------------------------------------------------------------------

def create_self_healing_dag():
    """
    Initializes and compiles the LangGraph state machine.
    """
    workflow = StateGraph(GraphState)

    # 1. Add the nodes
    workflow.add_node(INFERENCE_NODE, run_inference)
    workflow.add_node(CONFIDENCE_CHECK_NODE, check_confidence)
    workflow.add_node(FALLBACK_NODE, trigger_fallback)

    # 2. Set the entry point
    workflow.set_entry_point(INFERENCE_NODE)

    # 3. Define edges (path 1: straight through)
    workflow.add_edge(INFERENCE_NODE, CONFIDENCE_CHECK_NODE)

    # dag_pipeline.py (CORRECTED CODE)
    # 4. Define conditional edges (The decision point)
    workflow.add_conditional_edges(
        CONFIDENCE_CHECK_NODE,
        # The condition function extracts the routing key from the state
        lambda state: state["decision"],
        {
            "accept": END,
            "fallback": FALLBACK_NODE,
        },
    )

    # 5. Define fallback exit
    # In a real-world scenario, FallbackNode would lead to another step (like re-inference)
    # but for the CLI demo, it triggers user interaction and then ends the graph.
    # The CLI will handle the final logging after user input.
    workflow.add_edge(FALLBACK_NODE, END) 

    # 6. Compile the graph
    app = workflow.compile()
    
    log_event("DAG Initialization", "Self-Healing Classification DAG compiled successfully.")
    
    return app

# Initialize the DAG for export
self_healing_app = create_self_healing_dag()


