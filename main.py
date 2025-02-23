from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import logging
import uuid
from datetime import datetime

# PDF Text Extraction
import pypdf

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    get_response_synthesizer
)

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader

# from llama_index.llms import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import MetadataMode

from llama_index.core import Settings



# RagaAI
# from raga import RagTester, TestSuite, validate
# from raga.api.rag import LLMConfig, PromptTemplate

# MongoDB
import pymongo
from dotenv import load_dotenv

load_dotenv(override=True)
# Logging
logging.basicConfig(level=logging.INFO)

# Configuration (Move to environment variables in a real application)
MONGO_URI = os.getenv("MONGO_URI") #"mongodb://localhost:27017/"  # Replace with your MongoDB URI
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_DIR = os.getenv("INDEX_DIR")  # Directory to store the LlamaIndex vector index
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API Key
RAGAAI_API_KEY = os.getenv("RAGAAI_API_KEY") # Replace with your RagaAI API Key

app = FastAPI()

# MongoDB Client
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
conversation_collection = db[COLLECTION_NAME]

# LlamaIndex Setup
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
# service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Use Settings instead of ServiceContext
Settings.llm = llm
Settings.embed_model = embed_model

# RagaAI setup
#os.environ["RAGA_API_KEY"] = RAGAAI_API_KEY # Setting API key for raga
#rag_test = RagTester()
#prompt_template = PromptTemplate(
#        "You are a helpful assistant. Use the context provided to answer the query. {context_str} Query: {query_str}"
#    )
#llm_config = LLMConfig(model_name="gpt-3.5-turbo", prompt_template=prompt_template, open_ai_key = OPENAI_API_KEY)

#Data Models
class ChatbotRequest(BaseModel):
    message: str
    chat_id: str = None  # Optional:  If provided, continue the existing chat.

class ChatbotResponse(BaseModel):
    response: str
    chat_id: str
    metrics: Dict[str, Any] = None # For RagaAI, will contain evaluation metrics.

# Helper Functions
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extracts text content from a PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

def load_or_create_index(directory_path: str) -> VectorStoreIndex:
    """Loads an existing LlamaIndex or creates a new one from the given directory."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=directory_path)
        index = load_index_from_storage(storage_context, )
        logging.info("Loaded existing LlamaIndex from storage.")
        return index
    except FileNotFoundError:
        logging.info("Creating a new LlamaIndex.")
        documents = SimpleDirectoryReader(input_dir=directory_path).load_data()
        index = VectorStoreIndex.from_documents(documents, )
        index.storage_context.persist(persist_dir=directory_path)
        return index
    except Exception as e:
        logging.error(f"Error loading or creating index: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading or creating index: {e}")

def get_retriever(index: VectorStoreIndex, similarity_top_k: int = 2):
    """Build a retriever."""
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

def get_query_engine(index: VectorStoreIndex):
    """Build a query engine."""
    retriever = get_retriever(index)
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return query_engine

def get_or_create_chat_id() -> str:
    """Generates a unique chat ID."""
    return str(uuid.uuid4())

# RagaAI integration (commented out for now as it requires setup)
#def evaluate_chatbot_response(query: str, response: str) -> Dict[str, Any]:
#    """Evaluates the chatbot response using RagaAI."""
#    try:
#        run_id = str(uuid.uuid4())
#        data = {"query":query, "response":response, "run_id":run_id}
#        df = pd.DataFrame([data])
#        test_suite = TestSuite(
#            name="SproutsAI Chatbot Evaluation",
#            llm_config=llm_config,
#            source_data=df
#        )
#        test_suite.add_test(rag_test.context_relevance)
#        test_suite.run()
#        results = test_suite.calculate()
#        print(results.to_dict())
#        return results.to_dict() # Simplify for now; return all metrics
#    except Exception as e:
#        logging.error(f"Error evaluating chatbot response with RagaAI: {e}")
#        return {"error": str(e)} # Return error if evaluation fails

# API Endpoints
@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile):
    """Uploads a PDF, extracts text, and adds it to the knowledge base."""
    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        text = extract_text_from_pdf(pdf_file)
        print(text)
        # Save the text to a file in the data directory for LlamaIndex to process
        filename = pdf_file.filename
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filepath = os.path.join(data_dir, filename[:-4] + ".txt")  # Change extension to .txt

        with open(filepath, "w", encoding="utf-8") as f:  # Specify encoding
            f.write(text)

        # Rebuild the LlamaIndex
        index = load_or_create_index(directory_path="data")

        return {"filename": pdf_file.filename, "message": "PDF uploaded and processed successfully."}
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/chatbot/", response_model=ChatbotResponse)
async def chatbot(request: ChatbotRequest):
    """Handles chatbot requests, maintains chat history, and evaluates responses."""
    message = request.message
    chat_id = request.chat_id or get_or_create_chat_id()

    # Load LlamaIndex
    index = load_or_create_index(directory_path="data")
    query_engine = get_query_engine(index)

    try:
        #Query LLM
        response = query_engine.query(message)
        response_text = str(response) # Extract text from the LlamaIndex response object

        # Store conversation in MongoDB
        conversation_collection.insert_one({
            "chat_id": chat_id,
            "timestamp": datetime.utcnow(),
            "message": message,
            "response": response_text,
        })

        # Evaluate chatbot response using RagaAI (conditionally)
        #metrics = evaluate_chatbot_response(message, response_text) if RAGAAI_API_KEY else None
        metrics = None #Disabling raga AI

        return ChatbotResponse(response=response_text, chat_id=chat_id, metrics=metrics)

    except Exception as e:
        logging.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    app.run(host="0.0.0.0", port=8000)
