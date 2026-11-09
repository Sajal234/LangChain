from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('10.RAG/1.DocumentLoaders/intro.pdf')

docs = loader.load()

print(docs)

# len is equal to the number of pages in the pdf
print(f'\n{len(docs)}\n\n')

print(docs[0])


# there are different pdf loaders according to the need like PyPDFPlumberLoader(pdf with tables and columns), PyMuPDFLoader(with pdf form fields), PDFMinerLoader, UnstructuredFileLoader(scanned pdf, images), etc. 