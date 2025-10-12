import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Load the pre-trained model and the primary scaler ---
# These files should be in the same directory as this app.py file.
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    # This is a fallback for Gradio's deployment environments or if files are missing.
    model, scaler = None, None
    print("WARNING: Model or scaler files not found. The app will not work until a CSV is uploaded and a model is 'trained'.")


# --- 2. Define the Core Prediction Function ---
# This function takes the input values and returns the model's prediction.
def predict_fraud(*args):
    # The Gradio button passes all input fields as a tuple in *args
    # First, load the model and scaler from memory
    if model is None or scaler is None:
        return "Model not loaded. Please ensure model and scaler files are present."
        
    # Create a dictionary of feature names and their corresponding input values
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    input_values = dict(zip(feature_names, args))

    # Create a pandas DataFrame from the dictionary
    input_data = pd.DataFrame([input_values])

    # Preprocess the data using the loaded scaler
    input_data['Amount'] = scaler.transform(input_data[['Amount']])
    input_data['Time'] = scaler.transform(input_data[['Time']])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Return a user-friendly result
    if prediction == 1:
        return "🚨 Prediction: FRAUDULENT TRANSACTION"
    else:
        return "✅ Prediction: Legitimate Transaction"


# --- 3. Define Functions for UI Interactivity ---

# This function is triggered when a file is uploaded. It loads the CSV into a DataFrame.
def process_file(file):
    if file is not None:
        try:
            df = pd.read_csv(file.name)
            print("CSV file loaded successfully.")
            # Return the DataFrame to be stored in the gr.State component
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    return None

# This function fetches a random sample from the stored DataFrame.
def get_random_sample(df, fraud_class):
    if df is None:
        # If no data is loaded, return a list of 31 zeros (30 features + 1 class label)
        return [0] * 31
    
    # Filter the DataFrame for the desired class (0 for legitimate, 1 for fraud)
    sample_df = df[df['Class'] == fraud_class]
    if not sample_df.empty:
        # Select one random row and convert it to a list of values
        random_row = sample_df.sample(1).iloc[0]
        # The last value in the row is the 'Class' label
        expected_outcome = "FRAUDULENT" if random_row['Class'] == 1 else "Legitimate"
        # The first 30 values are the features to populate the input fields
        feature_values = random_row.drop('Class').tolist()
        # Return the feature values and the expected outcome string
        return feature_values + [expected_outcome]
    else:
        # If no samples of the requested class are found, return zeros and a message
        return [0] * 30 + [f"No samples of class {fraud_class} found in the uploaded file."]

# --- 4. Build the Gradio Interface with Blocks ---
# Using gr.Blocks() gives us more control over the layout and interactivity.
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # A gr.State component is used to store session-specific data, like our DataFrame
    stored_df = gr.State()

    gr.Markdown("# 💳 Enhanced Credit Card Fraud Detection Demo")
    gr.Markdown("Upload the `creditcard.csv` file, then click the buttons to fetch a random transaction and test the model's prediction.")

    with gr.Row():
        # Component to upload the CSV file
        file_upload = gr.File(label="Upload creditcard.csv", file_types=[".csv"])

    with gr.Row():
        # Buttons to trigger the auto-population of fields
        get_legit_button = gr.Button("Get Random Legitimate Transaction", variant="secondary")
        get_fraud_button = gr.Button("Get Random Fraudulent Transaction", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=3):
             # A list to hold all 30 number input components
            feature_inputs = [
                gr.Number(label="Time", interactive=True),
                gr.Number(label="V1", interactive=True), gr.Number(label="V2", interactive=True),
                gr.Number(label="V3", interactive=True), gr.Number(label="V4", interactive=True),
                gr.Number(label="V5", interactive=True), gr.Number(label="V6", interactive=True),
                gr.Number(label="V7", interactive=True), gr.Number(label="V8", interactive=True),
                gr.Number(label="V9", interactive=True), gr.Number(label="V10", interactive=True),
                gr.Number(label="V11", interactive=True), gr.Number(label="V12", interactive=True),
                gr.Number(label="V13", interactive=True), gr.Number(label="V14", interactive=True),
                gr.Number(label="V15", interactive=True), gr.Number(label="V16", interactive=True),
                gr.Number(label="V17", interactive=True), gr.Number(label="V18", interactive=True),
                gr.Number(label="V19", interactive=True), gr.Number(label="V20", interactive=True),
                gr.Number(label="V21", interactive=True), gr.Number(label="V22", interactive=True),
                gr.Number(label="V23", interactive=True), gr.Number(label="V24", interactive=True),
                gr.Number(label="V25", interactive=True), gr.Number(label="V26", interactive=True),
                gr.Number(label="V27", interactive=True), gr.Number(label="V28", interactive=True),
                gr.Number(label="Amount", interactive=True)
            ]
        with gr.Column(scale=1):
            # The main button to trigger the prediction
            predict_button = gr.Button("Flag Transaction", variant="primary")
            # Textboxes to display the expected and predicted outcomes
            expected_output = gr.Textbox(label="Expected Outcome")
            model_output = gr.Textbox(label="Model Prediction")

    # --- 5. Define the Component Interactions ---

    # When a file is uploaded, call the process_file function and store the result in stored_df
    file_upload.upload(fn=process_file, inputs=file_upload, outputs=stored_df)

    # When the "Get Legitimate" button is clicked...
    get_legit_button.click(
        fn=get_random_sample,
        # ...pass the stored DataFrame and the class label (0) as inputs...
        inputs=[stored_df, gr.Number(0, visible=False)],
        # ...and update the 30 feature inputs and the expected_output textbox with the results.
        outputs=feature_inputs + [expected_output]
    )
    
    # When the "Get Fraudulent" button is clicked...
    get_fraud_button.click(
        fn=get_random_sample, 
        # ...pass the stored DataFrame and the class label (1) as inputs...
        inputs=[stored_df, gr.Number(1, visible=False)],
        # ...and update the 30 feature inputs and the expected_output textbox with the results.
        outputs=feature_inputs + [expected_output]
    )

    # When the "Flag Transaction" button is clicked...
    predict_button.click(
        fn=predict_fraud,
        # ...pass all 30 feature inputs to the function...
        inputs=feature_inputs,
        # ...and display the result in the model_output textbox.
        outputs=model_output
    )

# --- 6. Launch the App ---
if __name__ == "__main__":
    demo.launch()