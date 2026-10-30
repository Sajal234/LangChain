from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


class Feedback(BaseModel):
    sentiment : Literal["Positive", "Negative"] = Field(description="The sentiment of the feedback")


parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions' : parser2.get_format_instructions()}
)


classifier_chain = RunnablePassthrough.assign(
    sentiment=prompt1 | model1 | parser2
)

prompt2 = PromptTemplate(
    template='You are a customer support agent. Reply directly to this positive customer review in 1-2 friendly sentences. Do not offer multiple options or explanations:\nCustomer Review: "{feedback}"\nYour Reply:',
    input_variables=['feedback'],
)
prompt3 = PromptTemplate(
    template='You are a customer support agent. Reply directly to this negative customer review in 1-2 polite and helpful sentences. Do not offer multiple options or explanations:\nCustomer Review: "{feedback}"\nYour Reply:',
    input_variables=['feedback'],
)

positive_chain = prompt2 | model1 | parser1
negative_chain = prompt3 | model1 | parser1


branch_chain = RunnableBranch(
    # format -  (condition1, chain1), (condition2, chain2), (default_chain)
    # works like if - elseif - elseif - else

    (lambda x:x['sentiment'].sentiment == 'Positive', positive_chain),
    (lambda x:x['sentiment'].sentiment == 'Negative', negative_chain),
    RunnableLambda(lambda x: "Could not find sentiment") # this is not a chain so we used Runnable Lambda to convert it into a chain 
)
 
    
chain = classifier_chain | branch_chain

response = chain.invoke({'feedback' : "This is a terrible smartphone"})

print(response)


chain.get_graph().print_ascii();