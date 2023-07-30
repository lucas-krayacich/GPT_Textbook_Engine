import streamlit as st
from langchain import OpenAI
import openai
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import tiktoken

# Textbook to be Searched
txt_name = "Sutton Reinforcement Learning textbook.pdf"
loader= UnstructuredPDFLoader(txt_name)
book_data = loader.load()

# # Split the raw text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
full_text = text_splitter.split_documents(book_data)

# # Create text vectors to represent each chunk
embeddings = OpenAIEmbeddings(openai_api_key="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")

# pinecone.init(
#     api_key="1003218d-8c9c-498f-9a59-94471f49f076",
#     environment="asia-southeast1-gcp-free"
# )
# index_name="langproj"

# # Store vectors in Pinecone Database
docsearch = Pinecone.from_texts([t.page_content for t in full_text], embeddings, index_name="langproj")

# query = "How does a markov chain work?"
# docs = docsearch.similarity_search(query)




