from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# import streamlit as st
from langchain.chains import ConversationChain

import tempfile
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain

from langchain_community.document_loaders import CSVLoader,PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
db_file_path='FAISS_vectorstores'
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if gemini_api_key is None:
    raise ValueError("API key not found in environment variables. Please check your .env file.")

# os.environ["OPENAI_API_KEY"] = openai_api_key

def pdf_loader(tmp_file_path):
    loader = PyPDFLoader(tmp_file_path)

    return loader

def csv_loader(tmp_file_path):
    loader = CSVLoader(tmp_file_path)

    return loader

# loadw = pdf_loader("data_results.txt")



# embeddings = HuggingFaceEmbeddings()

def creation_of_vectorDB_in_local(loaders):
  data = []
  for loader in loaders:
    data += loader.load()
    # print(data)

  db =FAISS.from_documents(data, embeddings)
  db.save_local(db_file_path)



def creation_FAQ_chain():
    db=FAISS.load_local(db_file_path, embeddings,allow_dangerous_deserialization=True)
    retriever =db.as_retriever(score_threshold=0.76)
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.8, convert_system_message_to_human=True)
    template = """
    You are an AI Question/Answer Chatbot.
    please response like a human you should interact in that way , for example if the query is hi then it will be much humanier way of answer
Given the following context and a question, answer the question as accurately and informatively as possible. 

Example 1:
Context: "The capital of France is Paris."
Human: "What is the capital of France?"
AI: "The capital of France is Paris."

Example 2:
Context: "The Eiffel Tower is located in Paris."
Human: "Where is the Eiffel Tower located?"
AI: "The Eiffel Tower is located in Paris."

Example 3:
Context: "The largest planet in our solar system is Jupiter."
Human: "Which is the largest planet in the solar system?"
AI: "The largest planet in our solar system is Jupiter."

Example 4:
Context: "Apple Inc. was founded in 1976."
Human: "When was Apple Inc. founded?"
AI: "Apple Inc. was founded in 1976."

Now, based on the context provided, answer the following question:
please give try to  answer
{context}
Current conversation: {chat_history}
Human: {query}
AI:
    """
    prompt = PromptTemplate(
        input_variables=["chat_history","query","context"],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key="chat_history",input_key="query",output_key='answer')


    chain = ConversationalRetrievalChain.from_llm(llm,
                                           retriever=retriever, 
                                           memory=memory,
                                           
                                           chain_type="stuff",
                                            get_chat_history=lambda h : h,
                                            combine_docs_chain_kwargs={'prompt': prompt},
                                           verbose=True,return_source_documents = True
                                          )
    return chain

def loading(docs):
    loaders = []
    for doc in docs:
        # print(doc)
        filename = doc.name 
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write the content of the UploadedFile to the temp file
            tmp_file.write(doc.read())
            tmp_file_path = tmp_file.name # Access the filename from UploadedFile object
        if filename.endswith('pdf'):
            load = pdf_loader(tmp_file_path )
        elif filename.endswith('csv') or filename.endswith('txt'):
            load = pdf_loader(tmp_file_path)  # This looks like it should be adjusted, is this correct for CSV and TXT?
        loaders.append(load)
    return loaders


# if _name_ == "_main_":

    

    # ans = creation_FAQ_chain()
    # con = 'You are a Data analyst you should provide the response as it was done by human'
    # while True:
    #     q = input()
    #     if q == 'Exit':
    #         break
    #     else:   
    #         res = ans({"context":con,"question":q,"query": q})
    #         # print(res)
    
    #         final_res = res['answer']
    #         print(final_res)
            
            
    # while True:
    #     query=input().strip()
    #     rebot(query)
    #     if query == "exit":
    #         break

