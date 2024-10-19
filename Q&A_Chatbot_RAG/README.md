# Q&A Chatbot with RAG for business Requirements

# Business Setup Question Answering (QA) Bot

## Overview

This project demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** model for a **Question Answering (QA)** bot. The QA bot is designed to answer questions related to setting up a business based on the document titled *Set Up Business*. It uses a vector database to store document embeddings and a generative model to generate coherent and informative answers.

## Features

- **Document Upload**: Users can upload business-related documents (PDFs) and ask questions based on the content.
- **Real-time Question Answering**: The bot retrieves relevant information from the document and generates answers to user queries.
- **Interactive UI**: The application is built with **Streamlit**, providing a user-friendly interface for document uploads and query inputs.
- **Chat History**: The system maintains the chat history of interactions, providing a seamless experience for the user.
  
## Technologies Used

- **Vector Database**: FAISS (or any other vector database) for storing and retrieving document embeddings.
- **Generative Model**: GEMINI API (or any other LLM) for generating responses based on retrieved context.
- **Streamlit**: Used for building the interactive frontend interface.
- **LangChain**: For managing the question-answering chain and retrieval process.

## Prerequisites

- Python 3.x
- A virtual environment (recommended)
- API access to a large language model (Gemini API, etc.)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Abubacker10/Generative-AI.git
   cd Q&A_Chatbot_RAG
2. **Create and activate a virtual environment:**
    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Set up API keys:**
   ```bash
   GOOGLE_API_KEY=your-gemini-api-key

6. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

8. **Upload the business document:**
   ```bash
   Upload the provided business document (set_up_business.pdf) through the interface.
   Ask questions in the chat interface, and the bot will respond based on the document content.

  ```bash
      ├── app.py                     # Main Streamlit app
      ├── chain.py                   # RAG model and document loading logic
      ├── requirements.txt           # Project dependencies
      ├── .env.example               # Example environment variables
      ├── README.md                  # Project README
      └── set_up_business.pdf        # Sample business document
      

This **README** template should provide all the necessary information for setting up and running your project. Let me know if you need further customization!
