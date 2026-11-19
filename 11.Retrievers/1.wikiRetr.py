import wikipedia
from langchain_community.retrievers import WikipediaRetriever


# wikipedia retriever fetches the data from wikipedia and returns the content

# Set a custom user-agent so Wikipedia API doesn't block you
wikipedia.set_user_agent("MyLangChainApp/1.0 (contact: test@example.com)")
retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of chinese"

docs = retriever.invoke(query)

# print(docs)

for i, doc in enumerate(docs):
    print(f"\n---Result {i+1} ---")
    print(f"Content : \n{doc.page_content}")