from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

messages = [
    SystemMessage(content="You are a helpful assistant. Answer questions based on the context provided."),
    HumanMessage(content="Tell me about langChain.")
]

response = model.invoke(messages)

text = response.content[0]["text"] if isinstance(response.content, list) else response.content

messages.append(AIMessage(content=text))

print("\nAI : "+ text)
print("\nMessages:")
print(messages)