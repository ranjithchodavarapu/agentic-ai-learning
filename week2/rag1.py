from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="KV-cache divergence occurs when cache-ON and cache-OFF inference paths produce different outputs due to FP16 non-associativity.", metadata={"source": "test1"}),
    Document(page_content="Greedy decoding failures happen when a model's argmax token selection leads to compounding errors over long generations.", metadata={"source": "test2"}),
    Document(page_content="RAG systems retrieve relevant document chunks and inject them into the LLM's context before generation.", metadata={"source": "test3"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

results = vectorstore.similarity_search("why does cache-off differ from cache-on?", k=2)
for doc in results:
    print(doc.page_content, "\n---")