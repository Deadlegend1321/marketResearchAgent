import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the embedding model globally
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
id2doc = {}


def initialize_rag_index(company_csv_path, trend_csv_path):
    global index, id2doc

    # Load your CSV data
    companies = pd.read_csv(company_csv_path)
    trends = pd.read_csv(trend_csv_path)

    # Format rows into searchable string documents
    company_docs = companies.apply(
        lambda r: f"Company: {r['CompanyName']} | Industry: {r['Industry']} | Desc: {r['Description']}",
        axis=1).tolist()
    trend_docs = trends.apply(lambda
                                  r: f"Industry: {r['Industry']} | Headline: {r['Headline']} | Snippet: {r['Snippet']} | Date: {r['Date']}",
                              axis=1).tolist()
    all_docs = company_docs + trend_docs

    # Map IDs to text
    id2doc = {i: doc for i, doc in enumerate(all_docs)}

    # Create vector embeddings
    embeddings = model.encode(all_docs, convert_to_numpy=True)

    # Create the FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    print(f"Database: Successfully indexed {len(all_docs)} documents.")


def rag_search(query, top_k=3):
    if index is None:
        raise ValueError("FAISS Index has not been initialized. Call initialize_rag_index first.")
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb).astype('float32'), top_k)
    return [id2doc[int(i)] for i in I[0]]