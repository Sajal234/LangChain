from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# gemini-3-flash is a chatModel not a LLM
llm = ChatGoogleGenerativeAI(model= "gemini-3-flash-preview")

response = llm.invoke("What is the capital of India")

print(response.content)



