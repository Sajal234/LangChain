from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import message
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of India and what are the best sites to get litematica house schemantics in Minecraft")

print(response.content)