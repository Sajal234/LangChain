from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv


load_dotenv()


# load the document file
loader = PyPDFLoader("/Users/sajal/Downloads/Resume/sajal_resume1.pdf")
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

# convert text into embeddings and store into FAISS
vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

# create a retriever (fetches the relevant document chunks based on the query)
retriever = vectorstore.as_retriever()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    
query = 'What are the key take aways from the document?'

response = qa_chain.run(query)
print(response)