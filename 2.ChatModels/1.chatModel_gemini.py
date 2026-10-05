from langchain_core.messages import HumanMessage
from langchain_core.messages.system import SystemMessage
from email import message
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0, max_completion_tokens=5)

message = [
    SystemMessage(content="You are a professional Minecraft player"),
    HumanMessage(content="which are the top 5 websites for litematica schematics design for a minecraft house or base")
]

response = model.invoke(message)

print(response.content[0]["text"])