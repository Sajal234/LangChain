from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(
        page_content="LangChain is an open-source framework designed to simplify the creation of applications powered by large language models (LLMs).",
        metadata={"topic": "LangChain"}
    ),
    Document(
        page_content="Chroma is an open-source AI-native embedding database and vector store built for fast similarity search and building LLM apps.",
        metadata={"topic": "Chroma"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by fetching relevant facts from external knowledge bases.",
        metadata={"topic": "RAG"}
    ),
    Document(
        page_content="Vector embeddings are numerical vector representations of text where semantically similar phrases are located close to each other in high-dimensional space.",
        metadata={"topic": "Embeddings"}
    ),
    Document(
        page_content="A Retriever in LangChain is an interface that takes an unstructured query as input and returns a list of relevant Document objects.",
        metadata={"topic": "Retrievers"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma.from_documents(
    embedding=embedding_model,
    documents=documents,
    collection_name="my_collection",
)


# vector_store can also do similarity search as retriever(in file 10.Rag/3.VectorStores/1.chromaDb.py) but it only does similarity search with only 1 strategy (default strategy).
# But retriever can use multiple strategies to search. Like similarity_search, mmr_search, etc. We can use any of them.
# For this reason, retriever is a better interface to work with.

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "what is chroma used for?"

response = retriever.invoke(query)

for i, doc in enumerate(response):
    print(f"\nResult {i+1}")
    print(f"{doc.page_content}")