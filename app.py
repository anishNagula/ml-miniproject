import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # Added this import

# --- 1. Load the trained model and scaler ---
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# --- 2. Define the prediction function ---
def predict_fraud(Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, 
                  V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, 
                  V21, V22, V23, V24, V25, V26, V27, V28, Amount):
    
    # Create a pandas DataFrame from the inputs
    input_data = pd.DataFrame({
        'Time': [Time], 'V1': [V1], 'V2': [V2], 'V3': [V3], 'V4': [V4],
        'V5': [V5], 'V6': [V6], 'V7': [V7], 'V8': [V8], 'V9': [V9],
        'V10': [V10], 'V11': [V11], 'V12': [V12], 'V13': [V13], 'V14': [V14],
        'V15': [V15], 'V16': [V16], 'V17': [V17], 'V18': [V18], 'V19': [V19],
        'V20': [V20], 'V21': [V21], 'V22': [V22], 'V23': [V23], 'V24': [V24],
        'V25': [V25], 'V26': [V26], 'V27': [V27], 'V28': [V28], 'Amount': [Amount]
    })

    # --- THIS IS THE CORRECTED PART ---
    # Scale BOTH 'Amount' and 'Time' using the SAME scaler loaded from your notebook
    input_data['Amount'] = scaler.transform(input_data[['Amount']])
    input_data['Time'] = scaler.transform(input_data[['Time']]) # Use the loaded scaler here too
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    # Return the result
    if prediction == 1:
        return "🚨 Prediction: FRAUDULENT TRANSACTION 🚨"
    else:
        return "✅ Prediction: Legitimate Transaction ✅"

# --- 3. Create the Gradio Interface ---
inputs = [
    gr.Number(label="Time (seconds since first transaction)"),
    gr.Number(label="V1"), gr.Number(label="V2"), gr.Number(label="V3"),
    gr.Number(label="V4"), gr.Number(label="V5"), gr.Number(label="V6"),
    gr.Number(label="V7"), gr.Number(label="V8"), gr.Number(label="V9"),
    gr.Number(label="V10"), gr.Number(label="V11"), gr.Number(label="V12"),
    gr.Number(label="V13"), gr.Number(label="V14"), gr.Number(label="V15"),
    gr.Number(label="V16"), gr.Number(label="V17"), gr.Number(label="V18"),
    gr.Number(label="V19"), gr.Number(label="V20"), gr.Number(label="V21"),
    gr.Number(label="V22"), gr.Number(label="V23"), gr.Number(label="V24"),
    gr.Number(label="V25"), gr.Number(label="V26"), gr.Number(label="V27"),
    gr.Number(label="V28"),
    gr.Number(label="Amount (Transaction Value)")
]

output = gr.Textbox(label="Model Prediction")

demo = gr.Interface(
    fn=predict_fraud, 
    inputs=inputs, 
    outputs=output,
    title="Credit Card Fraud Detection",
    description="Enter the transaction details to predict if it's fraudulent or legitimate. (All 30 fields are PCA components from V1-V28, plus Time and Amount)."
)

# --- 4. Launch the App ---
if __name__ == "__main__":
    demo.launch()
