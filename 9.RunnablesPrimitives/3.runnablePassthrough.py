from langchain_core.runnables import passthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


# # returns the input as output 
# passthrough = RunnablePassthrough()
# print(passthrough.invoke({'topic' : 'Minecraft'}))


model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1 = PromptTemplate(
    template='Tell me a joke about, {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke, {text}',
    input_variables=['text']
)

joke_generator =  RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model, parser)
})


# both 'joke' ans 'explanation'  run in parallel and receives the same input i.e. joke from the output of joke_generator
final_chain = RunnableSequence(joke_generator, parallel_chain)

response = final_chain.invoke({'topic' : 'Minecraft'})
print(response)