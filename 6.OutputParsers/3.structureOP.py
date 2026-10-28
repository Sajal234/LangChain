from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()



# in updated version structured output parser is replaced by json output parser along with pydantic
# and ResponseSchema is replaced by Pydantic Base Models




model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


# structure oputput parser helps in extraction of structured data from the model's response
# if the response contains some other data apart from the structured data, then it will extract only the structured data and return it.
# ResponseSchema is used to define the structure of the data we want to extract.

schema =[
    ResponseSchema(name = 'fact1', description = 'fact1 about the topic'),
    ResponseSchema(name = 'fact2', description = 'fact2 about the topic'),
    ResponseSchema(name = 'fact3', description = 'fact3 about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)


# we have to give instructions to the model to extract the data in a structured format.
# for that we use get_format_instructions()
# this function is specific to the type of output parser we are using.
template = PromptTemplate(
    template="Give three facts abut the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)


chain = template | model | parser

response = chain.invoke({'topic' : 'black hole'})

print(response)
