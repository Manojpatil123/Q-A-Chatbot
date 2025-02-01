import faiss
import uvicorn
import logging
from fastapi import FastAPI, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from fastapi.responses import PlainTextResponse
import json

# ✅ Initialize FastAPI
app = FastAPI(title="Q&A chatbot", version="1.0")

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Load FAISS Index and Metadata
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
# text_splitter = SemanticChunker(OpenAIEmbeddings())
hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
persisted_vectorstore = FAISS.load_local("faiss_index_constitution", hf_embeddings,allow_dangerous_deserialization=True)


# ✅ Initialize LLM (Change if using local LLM)
GROQ_API_KEY = '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'
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0
)
# ✅ Set up RetrievalQA Chain
qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=persisted_vectorstore.as_retriever(search_kwargs={"k": 5}),return_source_documents=True)

# 🔍 API Endpoint: Ask a Question
@app.get("/ask", summary="Ask a question and get an answer from bot")
def ask_question(query: str = Query(..., description="Enter your question")):
    """
    Query FAISS vector database and get a response from LLM.
    """
     try:
        logger.info(f"Received query: {query}")
        print(query)
        # Get AI-generated answer and source documents
        result = qa({"query": query}, return_only_outputs=True)

        # Appending source documents
        text = ""
        for doc in result.get('source_documents', []):
            text += doc.page_content

        logger.info(f"Generated answer: {result['result']}")

        response = {
            "query": query,
            "answer": result["result"],
            "retrieved_documents": text
        }
        return  result["result"]

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")

# ✅ Run API Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
