from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a joke on the topic, {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

parser = StrOutputParser()

def word_counter(text: str):
    return len(text.split())

joke_genertor = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'length' : RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_genertor, parallel_chain)

response = final_chain.invoke({'topic' : 'Minecraft'})

print(f' Joke : {response["joke"]}\n Word Count : {response["length"]}')