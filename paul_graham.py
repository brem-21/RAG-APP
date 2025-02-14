from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from uuid import uuid4

# Load environment variables
load_dotenv()

# Set up Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB client
chroma_db = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")
chroma_collection = chroma_db.get_or_create_collection("paul_graham", embedding_function=embedding_fn)

app = Flask(__name__)
CORS(app)

def max_token_length(txt_list: list):
    max_length = 0
    for txt in txt_list:
        token_count = len(re.findall(r'\w+', txt))
        if token_count > max_length:
            max_length = token_count
    return f"Max Token Length: {max_length} tokens"

model_max_chunk_length = 256
token_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=model_max_chunk_length,
    model_name="all-MiniLM-L6-v2",
    chunk_overlap=0
)

text_path = "paulgraham.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text_raw = f.read()

character_splitter = RecursiveCharacterTextSplitter(
    separators=['\n  \n', '\n\n', '\n', '. '],
    chunk_size=1000,
    chunk_overlap=0
)

text_splitted = character_splitter.split_text(text_raw)
max_token_length(text_splitted)

print(f"Total number of splitted chunks: {len(text_splitted)}")

text_tokens = []
for text in text_splitted:
    text_tokens.extend(token_splitter.split_text(text))

ids = [str(uuid4()) for _ in range(len(text_tokens))]
ids[:5]

chroma_collection.add(documents=text_tokens, ids=ids)

def get_query_results(query_text: str, n_results: int = 5) -> str:
    """Retrieve relevant information from ChromaDB."""
    res = chroma_collection.query(query_texts=[query_text], n_results=n_results)
    docs = res["documents"][0]

    # Check if 'metadatas' exists, is a list, and contains valid metadata dictionaries
    if res.get('metadatas') and isinstance(res['metadatas'], list) and len(res['metadatas']) > 0:
        keywords = [item.get('keyword', '') if isinstance(item, dict) else '' for item in res['metadatas'][0]]
    else:
        keywords = [''] * len(docs)  # Assign empty keywords if metadata is missing

    return '; '.join([f'{keyword}: {information}' for keyword, information in zip(keywords, docs)])

def rag_tool(user_query: str):
    """Retrieves relevant data from the database and generates a response using Gemini-2.0-Flash."""
    retrieved_results = get_query_results(user_query)
    
    system_prompt = (
        "You are an AI assistant with RAG capabilities. You will be given a user query and relevant retrieved documents. "
        "Please generate a response based only on the provided information."
    )

    full_query = f"User Query: {user_query}\n\nRetrieved Documents:\n{retrieved_results}\n\nInstruction: {system_prompt}"

    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        # Generate response
        response = model.generate_content(full_query)
        
        # Check if the response is valid and has text
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Unable to generate a response: Invalid response from the model."
    except Exception as e:
        # Log the error for debugging
        print(f"Error generating response: {e}")
        return f"Unable to generate a response: {str(e)}"

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    response = rag_tool(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)