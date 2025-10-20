import fitz
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from faiss import IndexFlatL2
import os

"""s
This script performs the embedding process for your documents.

Instructions:
- Insert your API key on this line: [GROQ_API_KEY = "YOUR_GROQ_API_KEY"]
- Insert the paths of the documents you want to embed into this array: pdf_files = []
- Don't forget to install all required dependencies.
- Run the code using: 'python rag.py'

IMPORTANT: Do not run this code!
The document copies used by the developer are not publicly shared.
This script primarily generates an output folder named faiss_index, so check whether it already exists in this repository.
However, you can still use this code to embed your own documents without any problem.
"""

# Write your Groq API key here
# If you don't have one, go to console.groq.com to get it
GROQ_API_KEY = "GROQ_API_KEY"

# Set the Groq API key in your current environment
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# List of your PDF files
pdf_files = [
    """STAND UP.pdf"""
]


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Initialize the text splitter (you can adjust chunk_size and chunk_overlap)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Array to store all chunks
all_docs = []

# Read each PDF file from the provided paths
# Split it into multiple chunks
# Convert each chunk into a LangChain Document
# Then store all documents into an array
for pdf_path in pdf_files:
    text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]
    all_docs.extend(docs)

print(f"Total chunks from all PDFs: {len(all_docs)}")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the FAISS index
index = IndexFlatL2(len(embedding_model.embed_query("hello world")))

# Initialize the vector database
vector_db = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Store all the documents into the vector database
uuids = [str(uuid4()) for _ in range(len(all_docs))]
doc_ids = vector_db.add_documents(documents=all_docs, ids=uuids)

# Save the local FAISS index
vector_db.save_local("faiss_index")

print("Embedding process completed successfully.")