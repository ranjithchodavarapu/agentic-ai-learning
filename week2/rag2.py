from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import glob
import shutil, os

pdf_files = glob.glob("papers/*.pdf")
all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    #print(f"Loading {pdf_file}...")
    #print(f"Number of pages: {loader.load()}")
    all_docs.extend(loader.load())   # one Document per PDF page

print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
chunks = splitter.split_documents(all_docs)

print(f"Split into {len(chunks)} chunks")
#print(f"First chunk: {chunks[0].page_content[:500]}...")
#print(f"Last chunk: {chunks[-1].page_content[:500]}...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_dir = "chroma_papers_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_papers_db")

#vectorstore2 = Chroma(persist_directory="./chroma_papers_db", embedding_function=embeddings)
#results = vectorstore2.similarity_search("what causes KV-cache divergence?", k=5)
#print(f"Found {len(results)} results for query 'what causes KV-cache divergence?'")

results = vectorstore.similarity_search("what causes KV-cache divergence?", k=3)
for doc in results:
    print(f"[{doc.metadata.get('source')}, page {doc.metadata.get('page')}]")
    #print(doc.page_content[:300])
    print(doc.page_content)   # full chunk, not truncated
    print("=" * 60)
    print("---")

print(vectorstore._collection.count())
