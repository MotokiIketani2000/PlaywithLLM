from dotenv import load_dotenv
load_dotenv()


import os
import streamlit as st
from PIL import Image
import  google.generativeai as genai

genai.configure(api_key=os.getenv("GENAI_API_KEY"))


model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mine_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


st.set_page_config(page_title="Multi invoice extracter")
st.header("Multi invoice extracter")
input = st.text_input("Input Prompt:", key="input")
uploaded_file = st.file_uploader("choose an image of the invoice...", type=["png", "jpg", "jpeg"])

image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me about the invoice")
input_prompt="""
You are expert in understanding invoices. We will upload a image as imvoice
and you will hace to answer any questions based on the uploaded invoice image
"""

# if submit button clicked
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The response is")
    st.write(response)






