import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Load Model and Scaler ---
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError:
    model, scaler = None, None
    print("WARNING: Model/scaler files not found. The app will not work until files are present.")

# --- 2. Core Prediction and Analysis Function ---
def predict_fraud_with_cost_analysis(*args):
    # Unpack all arguments
    cost_fp = args[-2]
    threshold = args[-1]
    feature_args = args[:-2]

    if model is None or scaler is None:
        return "Model not loaded.", ""

    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    input_dict = dict(zip(feature_names, feature_args))
    input_data = pd.DataFrame([input_dict])
    
    transaction_amount = input_dict['Amount'] # Get original amount for cost analysis

    # --- Preprocessing ---
    input_data_scaled = input_data.copy()
    input_data_scaled['Amount'] = scaler.transform(input_data[['Amount']].values)
    input_data_scaled['Time'] = scaler.transform(input_data[['Time']].values)

    # --- Prediction based on Threshold ---
    probability_fraud = model.predict_proba(input_data_scaled)[0][1]
    prediction = 1 if probability_fraud >= threshold else 0

    if prediction == 1:
        prediction_text = f"🚨 **Prediction: FRAUDULENT**\n(Probability: {probability_fraud:.2%}, Threshold: {threshold:.2%})"
    else:
        prediction_text = f"✅ **Prediction: Legitimate**\n(Probability: {probability_fraud:.2%}, Threshold: {threshold:.2%})"

    # --- UNIQUE FEATURE: Cost-Benefit Analysis ---
    prob_legit = 1 - probability_fraud
    cost_of_flagging = prob_legit * cost_fp
    cost_of_not_flagging = probability_fraud * transaction_amount

    recommendation = "Flag Transaction" if cost_of_flagging < cost_of_not_flagging else "Do Not Flag"
    
    analysis_text = f"""
    ### 💸 Cost-Benefit Analysis:
    * **Cost of False Positive (Review):** ${cost_fp:,.2f}
    * **Cost of False Negative (Transaction Amount):** ${transaction_amount:,.2f}
    ---
    * **Expected Cost of FLAGGING:**
        `(P(Legit) * Cost_FP)` = `{prob_legit:.2%} * ${cost_fp:,.2f} = **${cost_of_flagging:,.2f}**`
    * **Expected Cost of NOT FLAGGING:**
        `(P(Fraud) * Txn_Amount)` = `{probability_fraud:.2%} * ${transaction_amount:,.2f} = **${cost_of_not_flagging:,.2f}**`
    ---
    ### **Recommendation: {recommendation}**
    """
    
    return prediction_text, analysis_text

# --- Utility functions (Unchanged) ---
def process_file(file):
    if file is not None:
        try:
            df = pd.read_csv(file.name)
            print("CSV file loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    return None

def get_random_sample(df, fraud_class):
    if df is None:
        return [0] * 30 + ["Please upload the CSV first."]
    
    sample_df = df[df['Class'] == fraud_class]
    if not sample_df.empty:
        random_row = sample_df.sample(1).iloc[0]
        expected_outcome = "FRAUDULENT" if random_row['Class'] == 1 else "Legitimate"
        feature_values = random_row.drop('Class').tolist()
        return feature_values + [expected_outcome]
    else:
        return [0] * 30 + [f"No samples of class {fraud_class} found in the uploaded file."]

# --- 3. Gradio UI with Cost Analysis ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    stored_df = gr.State()

    gr.Markdown("# 💳 Fraud Detection as a Decision-Support Tool")
    gr.Markdown("This demo includes **Interactive Cost-Benefit Analysis** to recommend the most financially sound action.")

    with gr.Row():
        file_upload = gr.File(label="Upload creditcard.csv (Required for 'Get Random' buttons)", file_types=[".csv"])

    with gr.Row():
        get_legit_button = gr.Button("Get Random Legitimate Transaction", variant="secondary")
        get_fraud_button = gr.Button("Get Random Fraudulent Transaction", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=3):
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
            cost_fp_input = gr.Number(label="Cost of a False Positive ($)", value=5, info="Cost to investigate a wrongly flagged transaction.")
            
            threshold_slider = gr.Slider(
                minimum=0.01, maximum=0.99, step=0.01, value=0.5,
                label="Prediction Threshold",
                info="Adjust the sensitivity. Lower threshold = more fraud flags."
            )
            predict_button = gr.Button("Analyze Transaction", variant="primary")
            expected_output = gr.Textbox(label="Expected Outcome")

            # --- THE FIX IS HERE: Outputs are now inside the right-hand column ---
            model_output = gr.Markdown(label="Model Prediction")
            analysis_output = gr.Markdown(label="Cost-Benefit Analysis")

    # Event Handlers
    file_upload.upload(fn=process_file, inputs=file_upload, outputs=stored_df)

    get_legit_button.click(
        fn=get_random_sample,
        inputs=[stored_df, gr.Number(0, visible=False)],
        outputs=feature_inputs + [expected_output]
    )
    
    get_fraud_button.click(
        fn=get_random_sample, 
        inputs=[stored_df, gr.Number(1, visible=False)],
        outputs=feature_inputs + [expected_output]
    )

    predict_button.click(
        fn=predict_fraud_with_cost_analysis,
        inputs=feature_inputs + [cost_fp_input, threshold_slider],
        outputs=[model_output, analysis_output]
    )

if __name__ == "__main__":
    demo.launch()