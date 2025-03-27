# Chat with PDF using AWS Bedrock

## Project Overview
This project is a Streamlit-based web application that allows users to upload and chat with PDF documents using AWS Bedrock services. The system extracts text from PDFs, generates vector embeddings using Amazon Titan, stores them in FAISS for similarity search, and retrieves responses from LLMs (Claude and Llama3) based on user queries.

## Features
- Upload and process PDF documents
- Generate vector embeddings using Amazon Titan
- Store embeddings in FAISS for efficient retrieval
- Use Claude or Llama3 LLMs for answering user queries
- Interactive web UI built with Streamlit

## Prerequisites
Before running this project, ensure you have the following:
- **Python 3.8+** installed 3.11 in Venv
- **AWS Account** with access to Bedrock services
- **Boto3** configured with valid AWS credentials
- **FAISS**, **LangChain**, and **Streamlit** installed

## Installation
1. Create a virtual environment:
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up AWS credentials:
   ```bash
   aws configure
   ```
   Provide your AWS Access Key, Secret Key, and Region.

## Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and go to `http://localhost:8501/`

## Project Structure
```
aws-bedrock-chat-pdf/
│-- app.py               # Main application script
│-- data/                # Folder containing PDFs
│-- faiss_index/         # Stored vector index
│-- requirements.txt     # Required Python packages
│-- README.md            # Project documentation
```

## Usage
1. **Upload PDFs**: Place PDF files in the `data/` directory.
2. **Generate Vectors**: Click the "Vectors Update" button to process PDFs.
3. **Ask Questions**: Enter a question and choose Claude or Llama3 for responses.
4. **Retrieve Answers**: The system finds the most relevant context from PDFs and generates an answer.

## Troubleshooting
- **Pickle Deserialization Error**: If FAISS loading fails, add `allow_dangerous_deserialization=True` when calling `FAISS.load_local()`.
- **Run Error with Multiple Outputs**: Modify `qa.run()` to use `qa.invoke(query)["result"]` instead.

## Future Enhancements
- Support for more document formats (DOCX, TXT)
- Integration with other vector databases
- UI improvements and authentication features
