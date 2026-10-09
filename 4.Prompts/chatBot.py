from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.load import loads
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

with open('4.Prompts/template.json') as f:
    template = loads(f.read())

chain = template | model

print("AI chatBot initiated ---- type 'exit' to quit")
print("="*50)

chat_history = [
    SystemMessage(content='You are a helpful assistant.')
]

while True:
    user_input = input('You : ')
    if(user_input.lower().strip() == "exit"):
        break

    if not user_input.strip():
        continue

    chat_history.append(HumanMessage(content=user_input))
    response = chain.invoke({"user_input" : user_input, "chat_history" : chat_history})

    text = response.content[0]["text"] if isinstance(response.content, list) else response.content
    chat_history.append(AIMessage(content=text))
    print(f"\nAI : {text}\n")

print("\n================================================")
print('Chat History : \n', chat_history)

