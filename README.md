# 💳 Credit Card Fraud Detection & Decision Support Tool

Nagula Anish - PES2UG23CS358
Muskan Goenka - PES2UG23CS355

This project uses a machine learning model to detect fraudulent credit card transactions and presents the results in an interactive web interface built with Gradio.

Beyond simple classification, this application is designed as a decision-support tool. It incorporates a unique Cost-Benefit Analysis feature that recommends the most financially sound action (to flag or not flag a transaction) based on the potential costs associated with fraud and false positives.

---

### ~ Features

- Real-time Fraud Prediction: Classifies transactions as "Legitimate" or "Fraudulent" using a pre-trained Random Forest model.
- Interactive Threshold Tuning: A slider allows you to adjust the model's sensitivity (prediction threshold) to see how it impacts the outcome.
- Cost-Benefit Analysis: A unique feature that calculates the expected financial cost of both flagging and not flagging a transaction, providing a clear business-oriented recommendation.
- Sample Data Loading: Easily load random legitimate or fraudulent transactions from the dataset to test the model's performance.

---

### ~ Prerequisites

Before you begin, ensure you have the following installed:
- Git
- python 3.8+
- pip (Python's package installer)

---

### ~ Setup and Installation

Follow these steps to set up the project on your local machine.

1. <u><span style="color:green;">Clone the Repository</span></u>

    Open your terminal and clone the repository to a location of your choice.

    ```git clone https://github.com/anishNagula/ml-miniproject.git```

    ```cd ml-miniproject```

2. <u><span style="color:green;">Create and Activate a Virtual Environment</span></u>

    It's recommended to use a virtual environment to keep project dependencies isolated.
    ```
    # Create the virtual environment
    python -m venv .venv

    # Activate it (macOS/Linux)
    source .venv/bin/activate

    # Or, activate it (Windows)
    .venv\Scripts\activate
    ```

3. <u><span style="color:green;">Install Dependencies</span></u>

    This project uses several Python packages.
    - pandas
    - scikit-learn
    - gradio
    - numpy
    - joblib

    Install all the required packages by running:
    ```pip install -r requirements.txt```

4. <u><span style="color:green;">Download the Dataset</span></u>

    The creditcard.csv dataset is required for this project to function.

        Note: Due to its size, this file is typically not included in the git repository.

        - Download the dataset from Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud][https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud].

        - Place the downloaded creditcard.csv file in the same root directory as app.py.

---

### ~ How to Run and Use the Application

<u><span style="color:green;">Launch the App:</span></u>
Make sure your virtual environment is activated and you are in the project's root directory. Run the following command in your terminal:

```python app.py```

*Open the Web Interface:*

Your terminal will display a local URL, typically http://127.0.0.1:7860.

<u><span style="color:green;">Using the UI:</span></u>

<span style="color:purple;">*Upload the Dataset:*</span> The first step in the UI is to upload the `creditcard.csv` file using the file upload component.

<span style="color:purple;">*Load a Sample:*</span> Click "Get Random Legitimate Transaction" or "Get Random Fraudulent Transaction" to automatically fill all 30 feature fields with data.

<span style="color:purple;">*Set Business Rules:*</span>

Adjust the "Cost of a False Positive ($)" to reflect the operational cost of reviewing a transaction.

Move the "Prediction Threshold" slider to change the model's sensitivity.

<span style="color:purple;">*Analyze:*</span> Click the "Analyze Transaction" button.

<span style="color:purple;">*Review Results:*</span> The interface will update with two key outputs: the model's direct prediction on the right, and the detailed cost-benefit analysis and final recommendation below it.

