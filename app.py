import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Defining LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Function to load CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to load Excel file
def load_excel(file):
    return pd.read_excel(file, engine='openpyxl')

# Function to load JSON file
def load_json(file):
    return json.load(file)

def main():
    st.title("Data File Upload and Query Assistant")

    # File uploader for CSV, Excel, JSON
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx', 'json'])

    if uploaded_file is not None:
        # Check the file extension to handle appropriately
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            st.subheader("CSV File Preview")
            data = load_csv(uploaded_file)
        elif file_extension == 'xlsx':
            st.subheader("Excel File Preview")
            data = load_excel(uploaded_file)
        elif file_extension == 'json':
            st.subheader("JSON File Preview")
            data = load_json(uploaded_file)
            data = pd.json_normalize(data)  # Convert JSON to DataFrame for easier display
        else:
            st.error("Unsupported file type. Please upload a CSV, Excel, or JSON file.")
            return
        
        # Show a preview of the data
        st.write(data.head())  # Display first few rows of the data

        # Section for user to ask questions
        st.subheader("Ask a Question About the Dataset")
        user_query = st.text_input("Enter your query here:")
        
        if user_query:
            with st.spinner("Processing your query..."):
                # Generating a prompt based on user input
                prompt = f"Here is a preview of the dataset:\n{data.to_string(index=False)}\n\nUser's query: {user_query}"
                messages = [
                    (
                        "system",
                        "You are a helpful assistant that processes user queries by extracting relevant keywords from the prompt. These keywords are matched to the corresponding columns in the dataset shared by the user. Once matched, you return the relevant row or data from the dataset as per the user's request. Please ensure the responses are clear, precise, and actionable for the user."
                    ),
                    ("human", prompt),
                ]

                # Generate response
                ai_msg = llm.invoke(messages)
                st.write(f"LLM Response: {ai_msg.content}")

        # Allow the user to download the dataset (if required)
        

if __name__ == "__main__":
    main()
