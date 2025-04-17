
import streamlit as st
import pandas as pd
from transformers import pipeline
import datetime
import os

# Load a simple NLP pipeline (free HuggingFace model for demonstration)
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("AI-Powered Vinted Item Valuation & Sales Tracker")

# Section: AI Valuation
st.header("🛍️ Item Valuation")

item_description = st.text_area("Enter Item Description (brand, type, condition, etc.)")

if st.button("Estimate Item Value"):
    # Simplified valuation logic (this would typically be more complex or use regression)
    result = model(item_description)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    
    # Demo valuation logic based on sentiment for illustration
    if sentiment == "POSITIVE":
        estimated_value = 20 + (confidence * 30)  # Mock estimation
    else:
        estimated_value = 10 + (confidence * 10)

    st.write(f"Estimated Item Value: **£{estimated_value:.2f}**")

# Section: Sales Logging
st.header("📈 Sales Log")

# Check if sales log exists; create if not
log_file = "sales_log.csv"
if not os.path.isfile(log_file):
    df = pd.DataFrame(columns=["Date", "Item Description", "Sale Price (£)", "Buyer Info"])
    df.to_csv(log_file, index=False)

# Input fields for sales log
with st.form("log_sale_form"):
    sale_date = st.date_input("Date of Sale", datetime.date.today())
    sale_description = st.text_input("Sold Item Description")
    sale_price = st.number_input("Sale Price (£)", min_value=0.0, step=0.5)
    buyer_info = st.text_input("Buyer Information")

    submitted = st.form_submit_button("Log Sale")

    if submitted:
        new_sale = pd.DataFrame([[sale_date, sale_description, sale_price, buyer_info]],
                                columns=["Date", "Item Description", "Sale Price (£)", "Buyer Info"])
        
        df = pd.read_csv(log_file)
        df = pd.concat([df, new_sale], ignore_index=True)
        df.to_csv(log_file, index=False)
        st.success("Sale logged successfully!")

# View Sales History
if st.checkbox("Show Sales History"):
    df = pd.read_csv(log_file)
    st.dataframe(df)
