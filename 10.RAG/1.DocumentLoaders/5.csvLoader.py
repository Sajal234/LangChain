from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='10.RAG/1.DocumentLoaders/socialNetworkAds.csv')

docs = loader.load()


# makes one document obj for each row
print(len(docs))



# we can use lazy loading and can ask ques to a LLM and it will only load that much doc we asked

print(f'\n{docs[100].page_content}')