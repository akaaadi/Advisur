
import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def load_db(subject):
    db_path = f"vector_dbs/{subject}_db"
    if not os.path.exists(db_path):
        return None
    return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)


def generate_answer(subject, query):
    vectorstore = load_db(subject)
    if vectorstore is None:
        return "❌ Database not found.", 0

    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
    You are an expert tutor of {subject.upper()}.
    Use ONLY the given context to answer in long, simple, clear explanation.

    Context:
    {context}

    Question:
    {query}

    Detailed Answer:
    """

    llm = Ollama(model="llama3")
    answer = llm.invoke(prompt)

    ctx_emb = embedding_model.embed_query(context)
    ans_emb = embedding_model.embed_query(answer)
    accuracy = round(cosine_similarity([ctx_emb], [ans_emb])[0][0] * 100, 2)

    return answer, accuracy
