import streamlit as st
import pandas as pd
from transformers import pipeline
import datetime
import os
from PIL import Image

# NLP valuation model
text_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Image valuation model (using a free image classification model)
image_model = pipeline("image-classification", model="google/vit-base-patch16-224")

st.title("🛍️ AI-Powered Vinted Valuation & Sales Tracker")

# Section: AI Valuation via Text
st.header("📝 Valuation from Description")

item_description = st.text_area("Item Description (brand, type, condition, etc.)")

if st.button("Estimate Value from Description"):
    result = text_model(item_description)[0]
    sentiment = result['label']
    confidence = result['score']

    # Mock valuation logic
    if sentiment == "POSITIVE":
        estimated_value = 20 + (confidence * 30)
    else:
        estimated_value = 10 + (confidence * 10)

    st.write(f"**Estimated Value (from text):** £{estimated_value:.2f}")

st.markdown("---")

# Section: AI Valuation via Image
st.header("📷 Valuation from Image")

uploaded_image = st.file_uploader("Upload an image of the item:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Item", use_column_width=True)

    if st.button("Estimate Value from Image"):
        classification = image_model(image)[0]
        label = classification['label']
        score = classification['score']

        # Mock valuation logic based on image category confidence
        estimated_value_img = 15 + (score * 35)

        st.write(f"Identified as: **{label}** ({score:.2%} confidence)")
        st.write(f"**Estimated Value (from image):** £{estimated_value_img:.2f}")

st.markdown("---")

# Section: Sales Logging
st.header("📈 Sales Log")

log_file = "sales_log.csv"
if not os.path.isfile(log_file):
    df = pd.DataFrame(columns=["Date", "Item Description", "Sale Price (£)", "Buyer Info"])
    df.to_csv(log_file, index=False)

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

if st.checkbox("Show Sales History"):
    df = pd.read_csv(log_file)
    st.dataframe(df)
