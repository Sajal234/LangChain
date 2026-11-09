from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

loader = DirectoryLoader(
    path='/Users/sajal/Downloads/Resume',
    glob='*.pdf',         # to tell to only load .pdf files
    loader_cls=PyPDFLoader 
)




## if we has larger chunks of pdf then .load() loads all the chunks in memory at once
# docs = loader.load()
# print(len(docs))
# print(docs)

# print(f'{docs[0].page_content}') 



# to load chunks one by one we can use .lazy_load() or iterate over the loader and it doesnt load all the chunks in memory at once, instead it laods one by one
docs = loader.lazy_load()
for doc in docs:
    print(doc)              


