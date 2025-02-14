# RAG-Powered AI Chatbot

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that enhances AI-generated responses by integrating a document retrieval system. The chatbot leverages a combination of a vector database (ChromaDB) and a generative AI model (Google Gemini) to provide more accurate and context-aware responses based on stored knowledge.

## Features
- **Retrieval-Augmented Generation (RAG):** Combines information retrieval and AI-generated responses for improved accuracy.
- **ChromaDB Integration:** Uses a vector database for efficient document storage and retrieval.
- **Google Gemini AI:** Utilizes Gemini-2.0-Flash to generate intelligent responses based on retrieved data.
- **Text Processing:** Implements advanced text chunking and embedding techniques to optimize information retrieval.
- **REST API:** Exposes an API endpoint for querying and obtaining AI-driven responses.

## How It Works
1. **Data Ingestion:** The system processes a text corpus by splitting it into smaller chunks and embedding them into ChromaDB.
2. **Query Processing:** When a user submits a query, relevant documents are retrieved from ChromaDB based on semantic similarity.
3. **AI Response Generation:** The retrieved documents and user query are combined and sent to the Gemini AI model to generate a context-aware response.
4. **Response Delivery:** The AI-generated response is returned via a REST API.

## Requirements
- Python 3.x
- Flask
- Flask-CORS
- Google Generative AI SDK
- ChromaDB
- LangChain
- SentenceTransformers
- dotenv

## Use Cases
- AI-powered chatbots with knowledge retrieval capabilities
- Document-based Q&A systems
- Intelligent search assistants for research and documentation
- Enhancing AI models with domain-specific information

## Installation & Setup
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up environment variables, including the Google API key.
4. Run the Flask server with `python app.py`.

## API Endpoint
- **POST /query**
  - **Request:** `{ "query": "your question here" }`
  - **Response:** `{ "response": "AI-generated answer" }`

## Future Improvements
- Support for multiple document sources
- Enhanced text processing and ranking algorithms
- Deployment to cloud-based services for scalability

## License
This project is open-source and available under the MIT License.

Access the app on `http://34.240.0.57:5000/home`