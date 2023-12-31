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

# # Textbook to be Searched
txt_name = "Sutton Reinforcement Learning textbook.pdf"
# loader= UnstructuredPDFLoader(txt_name)
# book_data = loader.load()

# # # Split the raw text into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# full_text = text_splitter.split_documents(book_data)

# # Create text vectors to represent each chunk
embeddings = OpenAIEmbeddings(openai_api_key="")

pinecone.init(
    api_key="",
    environment="asia-southeast1-gcp-free"
)
index_name="langproj"

# # Store vectors in Pinecone Database
# docsearch = Pinecone.from_texts([t.page_content for t in full_text], embeddings, index_name="langproj")
docsearch = Pinecone.from_existing_index(index_name, embeddings)
# query = "How does a markov chain work?"
# docs = docsearch.similarity_search(query)

# initialize llm
llm= OpenAI(temperature=0, openai_api_key="")
chain = load_qa_chain(llm, chain_type="stuff")


# Front end Streamlit Page
st.set_page_config(page_title="Langchain Textbook Search", page_icon=":robot:")
st.markdown("# Langchain Textbook Search")

col1, col2 = st.columns(2)
with col1: 
    st.markdown("### Textbook In Use:")
    st.markdown(txt_name)

with col2:
    st.image(image='RL_text_image.png')

st.markdown("### Query This Book:") 
def get_text():
    input_text = st.text_area(label="", placeholder="Question for Textbook...", key="query_input")
    return input_text

query = get_text()



if query:
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    st.markdown("### Generated Result:")
    st.write(answer)
# query = "Give an example of a Reinforcement Learning Application"


