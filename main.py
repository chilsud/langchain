import os
import streamlit as st
import pickle
import time
from langchain_openai import AzureChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

endpoint = "https://dev-openai-service-01.openai.azure.com/"
model_name = "gpt-4o"
deployment = "b2grp4-e51444c1-62b6-4934-a875-d7d23fe25e53"
subscription_key = os.getenv("OPENAI_API_KEY")
api_version = "2024-12-01-preview"

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are an assistant. Answer the question using ONLY the provided context.

    Context:
    {context}

    Question:
    {input}

    Rules:
    - Cite the source filename for every fact you use
    - At the end, list ONLY the sources you actually used
    - If you did not use a source, do not list it
    """
)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        openai_api_key=subscription_key,
        temperature=0.9,
        max_tokens=500
   )

if process_url_clicked:

    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            chain = create_stuff_documents_chain(llm, prompt)
            qa_chain = create_retrieval_chain(retriever, chain)


            #chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            #result = chain({"question": query}, return_only_outputs=True)
            result = qa_chain.invoke({"input": query}, temperature=0.9)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])
            print("**********  Result is *************")
            print(result)

            # Display sources, if available
            source_docs = result.get("context", [])

            if source_docs:
                st.subheader("Sources:")
                for i, doc in enumerate(source_docs, 1):
                    source = doc.metadata.get("source", "Unknown source")
                    st.write(f"{i}. {source}")




