# LLM Zero-Shot Phishing Detection Practice

## Description

This project demonstrates a simple **binary text classification** task focused on **phishing detection**, using a **Large Language Model (LLM)** accessed via the Hugging Face `transformers` library's zero-shot classification pipeline. It attempts to classify short text snippets (like email subjects or excerpts) into 'phishing' or 'not phishing' categories using the `facebook/bart-large-mnli` model without specific fine-tuning.

The primary goal of this script is to fulfill requirements for an assignment focusing on:
1.  Using an LLM for a classification task (specifically, phishing detection as suggested).
2.  Practicing the use and interpretation of common classification evaluation metrics from `scikit-learn`.

The script performs the following steps:
1. Defines a small sample dataset of email snippets with phishing/not phishing labels.
2. Splits the data into a pseudo test set.
3. Loads the pre-trained zero-shot classification pipeline (`facebook/bart-large-mnli`).
4. Makes predictions on the test snippets.
5. Evaluates the predictions using standard metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report).
6. **Prints a summary of the obtained results and brief explanations of the `scikit-learn` metric functions used**, as requested by the assignment.

## Dataset Note

The text dataset used in this script is intentionally small (15 samples total) for simplicity and to focus on demonstrating the workflow of using an LLM pipeline with scikit-learn metric evaluation. Consequently, the test set size is also very small (e.g., 5 samples).

While this setup allows the script to run quickly and clearly illustrate the required concepts, please note that the evaluation results obtained (Accuracy, Precision, Recall, F1-score) are **based on too few samples to be statistically significant**. They serve as an illustration of the metric calculation process rather than a robust assessment of the LLM's true performance capabilities on phishing detection. Rigorous model evaluation requires testing on much larger, specialized datasets.

## Requirements

The script requires Python 3 and the following libraries:

* `numpy`
* `scikit-learn`
* `transformers`
* `torch` (or `tensorflow` as an alternative backend)

Specific versions are listed in the `requirements.txt` file. You will also need an internet connection when running for the first time to download the pre-trained `facebook/bart-large-mnli` model (~1.6GB).

## Setup (for Local Machine)

1.  **Clone the repository or download the files** into a local folder.

2.  **Navigate to the project folder** in your terminal or command prompt.

3.  **Create and activate a virtual environment** (Highly Recommended):
    ```bash
    # Create a virtual environment (e.g., named 'venv')
    python -m venv venv

    # Activate it:
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
    *Using a virtual environment prevents conflicts with other Python projects.*

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This command reads the `requirements.txt` file and installs the specified packages and versions into your active virtual environment.)*

## Usage

Once the setup is complete (and your virtual environment is activated), run the Python script from your terminal:

```bash
python llm_phishing_detection.py
```
*(Assuming you saved the Python script as `llm_phishing_detection.py`)*

**Important Notes for Local Execution:**
* **Model Download:** The first time you run the script, the `transformers` library will download the pre-trained model (`facebook/bart-large-mnli`). This requires an internet connection and may take several minutes depending on your connection speed. The model size is approximately 1.6GB.
* **Resource Usage:** LLM models can be computationally intensive. Performance will depend on your CPU/GPU. This model should run reasonably well on modern CPUs.
* **Cache:** Downloaded models are typically cached locally by the `transformers` library, so subsequent runs will be much faster.

## Output

The script will print the following to the console:

1. Progress updates for data preparation, pipeline loading, and prediction for the phishing detection task.
2. **A results summary section** reporting the actual Accuracy, Precision, Recall, and F1 Score achieved on the test set during that specific run.
3. The 2x2 Confusion Matrix for 'phishing'/'not phishing' classes.
4. The detailed Classification Report showing metrics per class.
5. **A section briefly explaining the purpose, inputs, and outputs of the main `scikit-learn` metric functions used** (Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report).

This output demonstrates how to apply and evaluate an LLM for a phishing detection task and directly includes the result summary and function explanations required for the assignment.

## File Structure

```
.
├── llm_phishing_detection.py  # The main Python script for phishing detection
├── requirements.txt           # List of required Python packages (unchanged)
└── README.md                  # This explanation file (UPDATED for phishing task)
```