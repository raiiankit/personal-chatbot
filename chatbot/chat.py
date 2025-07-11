import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DB_DIR = "chatbot/db"
EMBED_MODEL = "models/embedding-001"
GEN_MODEL = "models/gemini-1.5-flash-latest"

# --- Helper to embed a question using Gemini
def get_embedding(text, task="retrieval_document"):
    kwargs = {
        "model": EMBED_MODEL,
        "content": text,
        "task_type": task,
    }

    if task == "retrieval_document":
        kwargs["title"] = "PersonalData"

    return genai.embed_content(**kwargs)["embedding"]

# --- Load vector DB
db = FAISS.load_local(DB_DIR, embeddings=None, allow_dangerous_deserialization=True)


# --- Custom Gemini-compatible class for query embedding
class GeminiEmbedding:
    def embed_query(self, text):
        return get_embedding(text)

embedding_model = GeminiEmbedding()

# --- Gemini chat model
gen_model = genai.GenerativeModel(GEN_MODEL)

# --- Main loop
print("\n🧠 Personal Chatbot is ready! Ask me anything about yourself.")
print("Type 'exit' to quit.\n")

while True:
    query = input("💬 You: ").strip()
    if query.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # Embed and retrieve relevant docs
    query_embedding = get_embedding(query, task="retrieval_query")
    docs = db.similarity_search_by_vector(query_embedding, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate answer using Gemini
    prompt = f"""
    Answer the question using the following context (if relevant). If it isn't in the context, say you don't know.

    Context:
    {context}

    Question: {query}
    """

    try:
        response = gen_model.generate_content(prompt)
        print(f"🤖 AI: {response.text.strip()}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
