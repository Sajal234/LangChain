from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


doc1 = Document(
    page_content="Virat Kohli is one of the finest modern-day batsmen, famous for his unmatched chasing masterclasses, incredible fitness, and leading the all-time scoring charts in international cricket.",
    metadata={"team": "Royal Challengers Bengaluru"}
)

doc2 = Document(
    page_content="MS Dhoni is revered as one of cricket's sharpest tactical minds and greatest match-finishers, having captained India to triumphs in the T20 World Cup, ODI World Cup, and Champions Trophy.",
    metadata={"team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Rohit Sharma, known as 'Hitman', is an explosive opening batsman who holds the world record for the highest individual ODI score of 264 runs and has captained his franchise to 5 IPL titles.",
    metadata={"team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is a world-class fast bowler renowned for his unorthodox bowling action, pinpoint yorkers in the death overs, and ability to extract bounce on any pitch.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Rishabh Pant is a dynamic left-handed wicketkeeper-batsman celebrated for his fearless stroke play and game-winning counter-attacking knocks in Test cricket across Australia and England.",
    metadata={"team": "Delhi Capitals"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory='chroma_db',
    collection_name='sample'
)

## commented this bcz we are not adding new documents to the vector store again and again
# vector_store.add_documents(docs)

data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
# print(data)



# there is one more function similarity_search_with_score, which returns the similarity score along with the documents
best_captain = vector_store.similarity_search(
    query='Who is the best captain',
    k=1,  # k tells us number of documents to be retrieved based on similarity
    # filter={"team" : "Mumbai Indians"}  # filter is used to filter the documents based on metadata (like where team == Mumbai Indians)
)
print(f'''Best Captain : {best_captain}''')



updated_doc1 = Document(
    page_content="Virat Kohli is one of the finest modern-day batsmen, famous for his unmatched chasing masterclasses, incredible fitness, and leading the all-time scoring charts in international cricket. He is also the former captain of the Indian cricket team.",
    metadata={"team": "Royal Challengers Bengaluru"}
)

vector_store.update_document(
    document_id="b3abb86e-c9a5-4cd8-9170-3531ceee928a",
    document=updated_doc1
)

print(f'''\n\nUpdated document : {updated_doc1}''')


# to delete documents
vector_store.delete(ids=["b3abb86e-c9a5-4cd8-9170-3531ceee928a"])