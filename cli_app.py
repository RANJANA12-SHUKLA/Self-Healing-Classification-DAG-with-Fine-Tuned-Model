# cli_app.py

import sys
from dag_pipeline import self_healing_app, GraphState
from utils.config import (
    LABEL_MAP, CONFIDENCE_THRESHOLD, INFERENCE_NODE, 
    CONFIDENCE_CHECK_NODE, FALLBACK_NODE, CORRECTED_TAG
)
from utils.logger import log_event, logging

def run_cli_loop():
    """
    Main loop for the Command Line Interface (CLI).
    Handles user input, runs the LangGraph, and manages the fallback interaction.
    """
    print("\n" + "="*50)
    print("Self-Healing Classification Pipeline (LangGraph)")
    print(f" Threshold for Fallback: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(" Enter 'quit' or 'exit' to stop the application.")
    print("="*50 + "\n")

    # The LangGraph app is compiled in dag_pipeline.py and imported here
    app = self_healing_app
    
    # Keep track of the run number for logging clarity
    run_count = 1 

    while True:
        try:
            # 1. Get Initial Input
            user_input = input(f"[{run_count}] Enter text for classification: \n> ")

            if user_input.lower() in ['quit', 'exit']:
                log_event("CLI Exit", "Application shutting down by user request.")
                print("\nShutting down. Check 'logs/run_log.txt' for the full history.")
                break
            
            if not user_input.strip():
                print("Please enter some text.")
                continue
            
            log_event("CLI Input", f"New request from user (Run {run_count}): '{user_input}'")

            # 2. Initialize and Run the Graph
            initial_state: GraphState = {
                "text_input": user_input,
                "prediction": "",
                "confidence_score": 0.0,
                "final_label": "",
                "user_clarification": "",
                "step_history": []
            }
            
            # Use stream to see the nodes execute, but we'll capture the final state
            # Note: LangGraph's stream is often used for real-time output, but here 
            # we run it to completion to get the final state cleanly.
            final_state = app.invoke(initial_state)

            # 3. Handle Output and Fallback Logic
            
            # Check the step history to see if the fallback node was activated
            if FALLBACK_NODE in final_state["step_history"]:
                # --- FALLBACK ACTIVATED (Human-in-the-loop required) ---
                print("\n" + "="*50)
                log_event("CLI Interaction", "FALLBACK MODE: Initiating Human-in-the-Loop.")
                print(f"[FallbackNode] Could you clarify your intent? Initial guess was '{final_state['prediction']}'.")
                
                # Get clarification from the user
                clarification = input("User clarification (e.g., 'Yes, negative' or 'No, it was positive'):\n> ")
                
                # Log the user interaction
                log_event("CLI Interaction", f"User clarification received: '{clarification}'")
                
                # Simple correction logic: If user mentions 'negative' or 'positive', correct the label
                # In a real system, you'd use a backup model or more complex NLP here.
                clarification_lower = clarification.lower()
                
                # --- Decision Based on Clarification ---
                if 'negative' in clarification_lower and final_state['prediction'] != LABEL_MAP[0]:
                    final_label = LABEL_MAP[0] # Negative (0)
                elif 'positive' in clarification_lower and final_state['prediction'] != LABEL_MAP[1]:
                    final_label = LABEL_MAP[1] # Positive (1)
                else:
                    # If clarification is ambiguous or doesn't suggest a change, trust the initial guess
                    final_label = final_state['prediction']
                
                # --- Final Logging & Output ---
                log_event("Final Decision", f"Final Label: {final_label} {CORRECTED_TAG}")
                print(f"\nFinal Label: {final_label} {CORRECTED_TAG}")
                print("="*50 + "\n")
                
            else:
                # --- ACCEPTED (High Confidence) ---
                final_label = final_state.get("final_label") or final_state.get("prediction", "N/A")
                confidence = final_state.get("confidence_score", 0.0)

                log_event("Final Decision", f"Final Label: {final_label} (Confidence: {confidence * 100:.1f}%)")
                print("\n" + "="*50)
                print(f"[Accepted] Final Label: {final_label}")
                print(f"Confidence: {confidence * 100:.1f}%")
                print("="*50 + "\n")

            
            run_count += 1
            
        except Exception as e:
            log_event("CLI Error", f"An unexpected error occurred: {e}", level=logging.ERROR)
            print(f"\n[ERROR] An error occurred. See logs/run_log.txt for details. Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_cli_loop()