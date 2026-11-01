from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
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

# manually retrieve relevant documents
query = 'What are the key take aways from the document?'
retrieved_docs = retriever.invoke(query)

# combined retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])


model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0.7)

# manually passed retrieved to LLM
prompt = PromptTemplate(
    template='Based on following text, answer the question: \n {retrieved_text} \n Question: {query}',
    input_variables=['retrieved_text','query']
) 

parser = StrOutputParser()

chain = prompt | model | parser

answer = chain.invoke({'retrieved_text' : retrieved_text, 'query' : query})

print(answer)