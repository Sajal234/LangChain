from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()


# Instead of hoping the user typed the "perfect query", MultiQueryRetriever uses an LLM to rephrase the query from multiple perspectives, runs all of them, and unions the results.

documents = [
    Document(
        page_content="Practicing deep breathing exercises and daily meditation can significantly lower elevated cortisol levels and calm an overactive nervous system.",
        metadata={"id": 1, "topic": "stress_management"}
    ),
    Document(
        page_content="Consuming a balanced diet rich in leafy greens, omega-3 fatty acids, and whole grains enhances gut health and stabilizes mood fluctuations.",
        metadata={"id": 2, "topic": "nutrition"}
    ),
    Document(
        page_content="Aerobic workouts such as jogging, cycling, and swimming release mood-boosting endorphins and promote sound nocturnal sleep quality.",
        metadata={"id": 3, "topic": "fitness"}
    ),
    Document(
        page_content="Cognitive behavioral techniques and journaling offer practical ways to manage persistent worry, tension, and acute anxiety symptoms.",
        metadata={"id": 4, "topic": "mental_health"}
    ),
    Document(
        page_content="Staying hydrated with 2 to 3 liters of water per day improves metabolic efficiency, cognitive sharpness, and physical energy levels throughout the afternoon.",
        metadata={"id": 5, "topic": "hydration"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

similarity_retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': 2
    }
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={'k':2}),
    llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
)

query = "I'm feeling stressed and anxious, what should I do?"


similarity_response = similarity_retriever.invoke(query)
multi_query_response = multi_query_retriever.invoke(query)

for i, doc in enumerate(similarity_response):
    print(f"\nSimilarity Response {i}", doc.page_content)

print("\n\n")
for i, doc in enumerate( multi_query_response):
    print(f"\nMulti Query Response {i}", doc.page_content)