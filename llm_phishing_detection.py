# llm_phishing_detection.py

# Import necessary libraries
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
import os

# Suppress specific warnings (optional)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Using a pipeline without specifying a model.*")
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

def run_llm_phishing_detection_task():
    """
    Demonstrates binary text classification for Phishing Detection using an
    LLM zero-shot pipeline and evaluates results using scikit-learn metrics.
    """
    print("--- Starting LLM Zero-Shot Phishing Detection Task ---")

    # --- 1. Define Data (Illustrative Phishing/Not Phishing Examples) ---
    # Using a minimal dataset for demonstration
    # These could be email subjects or short snippets.
    texts = [
        "Meeting Reminder: Project Update tomorrow at 10 AM", # Not Phishing (0)
        "URGENT: Your account has been compromised! Click here to secure.", # Phishing (1)
        "Weekly newsletter - Updates and Insights", # Not Phishing (0)
        "Verify Your PayPal Account Now - Security Alert!", # Phishing (1)
        "Lunch plans for Friday?", # Not Phishing (0)
        "Invoice #INV-12345 from SecurePay Solutions", # Not Phishing (0)
        "Congratulations! You've won a $1000 Gift Card! Claim Now!", # Phishing (1)
        "Re: Question about your recent order", # Not Phishing (0)
        "Action Required: Unusual Sign-in Activity Detected on your Account", # Phishing (1)
        "Team Outing Details - Please RSVP", # Not Phishing (0)
        "Your Amazon Package Delivery Notification - Click to Track", # Phishing (1) - Common scam subject
        "Catching up soon?", # Not Phishing (0)
        "FWD: Funny Cat Video", # Not Phishing (0)
        "Final Notice: Your Subscription Requires Immediate Payment Update", # Phishing (1)
        "Review Your Recent Bank Statement Online", # Not Phishing (0)
    ]
    # Corresponding binary labels (0: not phishing, 1: phishing)
    labels = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]

    # Define candidate labels for the LLM for this task
    candidate_labels = ['not phishing', 'phishing']
    target_names = ['not phishing', 'phishing'] # For consistency in reports
    y_true_text = [target_names[label] for label in labels] # True labels as strings

    print("\n[Data Preparation]")
    print(f"Using a small example dataset with {len(texts)} email snippets.")
    print(f"Task: Classify snippets as '{candidate_labels[0]}' or '{candidate_labels[1]}'.")

    # Split data
    if len(texts) > 1:
        try:
            X_train_text, X_test_text, y_train_true_text, y_test_true_text = train_test_split(
                texts, y_true_text, test_size=0.3, random_state=42, stratify=y_true_text
            )
        except ValueError:
             X_train_text, X_test_text, y_train_true_text, y_test_true_text = train_test_split(
                texts, y_true_text, test_size=0.3, random_state=42
            )
    else:
        X_train_text, X_test_text, y_train_true_text, y_test_true_text = [], texts, [], y_true_text

    print(f"Test set size: {len(X_test_text)} samples")
    print("-" * 40)

    # --- 2. Load LLM Pipeline (Zero-Shot Classification) ---
    print("[Load LLM Pipeline]")
    try:
        model_name = "facebook/bart-large-mnli" # Can use the same model
        classifier = pipeline("zero-shot-classification", model=model_name)
        print(f"Zero-shot classification pipeline loaded (Model: {model_name}).")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("\nPlease ensure internet connection and required libraries installed.")
        return
    print("-" * 40)

    # --- 3. Make Predictions ---
    print("[Prediction]")
    if not X_test_text:
        print("Test set is empty.")
        return

    try:
        raw_predictions = classifier(X_test_text, candidate_labels, multi_label=False)
        y_pred_text = [pred['labels'][0] for pred in raw_predictions]
        print(f"Predictions made on {len(X_test_text)} test samples.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    print("-" * 40)

    # --- 4. Evaluate the Model using Metrics ---
    print("[Model Evaluation]")
    if not y_pred_text or len(y_test_true_text) != len(y_pred_text):
        print("Prediction output is missing or mismatched. Cannot evaluate.")
        return

    # Calculate metrics, specifying 'phishing' as the positive class
    accuracy = accuracy_score(y_test_true_text, y_pred_text)
    precision = precision_score(y_test_true_text, y_pred_text, pos_label='phishing', zero_division=0)
    recall = recall_score(y_test_true_text, y_pred_text, pos_label='phishing', zero_division=0)
    f1 = f1_score(y_test_true_text, y_pred_text, pos_label='phishing', zero_division=0)
    cm = confusion_matrix(y_test_true_text, y_pred_text, labels=candidate_labels)
    report = classification_report(y_test_true_text, y_pred_text, labels=candidate_labels, target_names=target_names, zero_division=0)

    # --- 5. Report Results and Explain Metrics (Including Disclaimer) ---
    print("[Results Summary - Phishing Detection]")
    print(f"The LLM zero-shot model achieved the following scores on the test set:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision (for 'phishing'): {precision:.4f}") # Precision for the positive class
    print(f"- Recall (for 'phishing'): {recall:.4f}")    # Recall for the positive class
    print(f"- F1 Score (for 'phishing'): {f1:.4f}")     # F1 for the positive class

    # Disclaimer about small test set
    print("\n*Important Note on Evaluation:*")
    print(f"* These metrics are based on a very small test set ({len(X_test_text)} samples) due to the minimal example dataset.")
    print("* Results primarily demonstrate the calculation process and are not statistically robust indicators of general model performance.*")

    print("\n*Note: Zero-shot performance varies based on model suitability, text complexity, and label clarity.*")


    print("\nConfusion Matrix:\n")
    print("          Predicted")
    print("         ", "  ".join(candidate_labels))
    print("True")
    # Indices match candidate_labels order ['not phishing', 'phishing'] -> [0, 1]
    print(f"{candidate_labels[0]:<12} {cm[0]}") # True Negative [0,0], False Positive [0,1]
    print(f"{candidate_labels[1]:<12} {cm[1]}") # False Negative[1,0], True Positive [1,1]


    print("\nClassification Report:\n")
    print(report)
    print("-" * 40)

    # --- Brief Explanation of Metric Functions Used ---
    # (This section remains the same as it explains the general functions)
    print("[Metric Function Explanations]")
    print("Main scikit-learn.metrics functions used:")
    print("\n1. accuracy_score:")
    print("   - Purpose: Computes the overall fraction of predictions that matched the true labels.")
    print("   - Inputs: y_true (true labels), y_pred (predicted labels).")
    print("   - Output: Accuracy score (float between 0.0 and 1.0).")
    print("\n2. precision_score:")
    print("   - Purpose: Measures accuracy of positive predictions (TP / (TP + FP)). Answers: Of those predicted positive, how many truly are?")
    print("   - Inputs: y_true, y_pred. 'pos_label' specifies the positive class in binary tasks.")
    print("   - Output: Precision score (float).")
    print("\n3. recall_score:")
    print("   - Purpose: Measures ability to find all actual positives (TP / (TP + FN)). Answers: Of all actual positives, how many were found?")
    print("   - Inputs: y_true, y_pred. 'pos_label' specifies the positive class.")
    print("   - Output: Recall score (float).")
    print("\n4. f1_score:")
    print("   - Purpose: Harmonic mean of precision and recall (balances both). Useful especially if class distribution is uneven.")
    print("   - Inputs: y_true, y_pred. 'pos_label' specifies the positive class.")
    print("   - Output: F1 score (float).")
    print("\n5. confusion_matrix:")
    print("   - Purpose: Creates a matrix showing counts of prediction outcomes (True Positives, True Negatives, False Positives, False Negatives).")
    print("   - Inputs: y_true, y_pred. 'labels' can enforce class order.")
    print("   - Output: NumPy array (the matrix).")
    print("\n6. classification_report:")
    print("   - Purpose: Generates a text summary of precision, recall, F1-score, and support (number of true instances) for each class.")
    print("   - Inputs: y_true, y_pred. 'target_names' provides display names for classes; 'labels' ensures order.")
    print("   - Output: Formatted string report.")
    print("-" * 40)
    print("--- Phishing Detection Task Complete ---")

# Ensure the main function runs only when the script is executed directly
if __name__ == "__main__":
    run_llm_phishing_detection_task()