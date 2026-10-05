from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# single query
query = "What is Litematica?"
query_vector = embedding.embed_query(query)
print(f"Query vector length: {len(query_vector)}")






# document query
documents = [
    "Minecraft is the best game ever", 
    "Litematica is a Minecraft mod that allows you to build structures in the game"
]
embeds = embedding.embed_documents(documents)

# Print the number of vectors generated and vector dimensions
print(f"Generated {len(embeds)} embeddings.")
print(f"Vector dimension (length of first vector): {len(embeds[0])}")
print(str(embeds))



