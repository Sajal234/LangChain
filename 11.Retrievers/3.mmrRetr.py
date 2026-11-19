from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(
        page_content="Python is widely used in Machine Learning and Data Science with libraries like NumPy, Pandas, and PyTorch.",
        metadata={"id": 4, "topic": "machine_learning"}
    ),
    Document(
        page_content="Python web development is powered by popular frameworks such as Django, Flask, and FastAPI.",
        metadata={"id": 5, "topic": "web_dev"}
    ),
    Document(
        page_content="Python features automatic garbage collection, dynamic typing, and a clean, readable syntax.",
        metadata={"id": 6, "topic": "features"}
    ),
]


embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = FAISS.from_documents(
    embedding=embedding_model,
    documents=documents,
)



# MMR (Maximal Marginal Relevance): Optimizes for both relevance AND diversity:
# Finds documents that are relevant to your query.
# Penalizes documents that are too similar to the ones already selected.
# Ensures the returned results cover different angles/aspects instead of repeating the same information.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5}          # lambda is responsible for diverse answers(at 1 it behaves like normal similarity search)
)


query = "Tell me about Python and it's uses"
response = retriever.invoke(query)

for i, doc in enumerate(response):
    print(f"\n---Result {i+1}---")
    print(f"{doc.page_content}")
    print(f"Metadata: {doc.metadata}")