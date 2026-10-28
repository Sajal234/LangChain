from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person, must be greater than 18')
    job: str = Field(description='Job of the person')
    city: str = Field(description='Name of city where the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate name, age, job and city of a fictional {place} person. \n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'place':'Indian'})
print("\n" + "="*50 + "\n")
print(prompt.text)
print("\n" + "="*50 + "\n")
    

chain = template | model | parser

response = chain.invoke({'place':'Indian'})

print(response)