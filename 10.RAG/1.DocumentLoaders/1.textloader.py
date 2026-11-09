from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt = PromptTemplate(
    template='Write a summary for the following poem: {poem}',
    input_variables=['poem']
)

chain = prompt | model | parser

loader = TextLoader('10.RAG/1.DocumentLoaders/minecraft.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))

doc = docs[0]

print(doc)
print(type(doc))

response = chain.invoke({'poem' : docs[0].page_content})
print('----------------------------------------------------------------------------------------------------')
print(f"Type of response : {type(response)}")
print(f'\nResponse : \n{response}')