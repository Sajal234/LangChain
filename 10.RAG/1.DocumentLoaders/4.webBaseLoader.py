from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt = PromptTemplate(
    template='Answer the following question\n {ques}, from the given text\n {text}',
    input_variables=['ques', 'text']
)

chain = prompt | model | parser


# we can pass multiple url's by passing them in a list
url = 'https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdhe4hn-a/p/itm8505e2f874525?pid=COMHZQX47WQ7YJZP&lid=LSTCOMHZQX47WQ7YJZPBZOHQJ&marketplace=FLIPKART&cmpid=content_computer_24004169905_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,815466423545,,,,c,,,,,,,&entryMethod=24004169905&&cmpid=content_24004169905_gmc_pla&gad_source=1&gad_campaignid=24004169905&gbraid=0AAAAADxRY58O6xbmaY-mSWBjF2b5yRakA&gclid=Cj0KCQiA-8XXBhDaARIsADnEOrC5OUDqdCHZLTtge_7bGORsdrWPLvmDwnhsYTEhZGLHO8G0sQxvuu4aAovoEALw_wcB'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(f'{docs[0].page_content}')


response = chain.invoke({'ques':'What is the product that we are talking about, specially mention its brand value', 'text':docs[0].page_content})

print(f'\n\nResponse: {response}')