from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = JsonOutputParser()


#format_instruction :- while making prompt we send additional instructions to the model to make sure it returns the output in the desired format.
#This format instruction is specific to the type of output parser we are using.
template = PromptTemplate(
    template="Give me name, age and city of a fictional \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.format()
#OR
#prompt = template.invoke({})

# In invoke we can also use {input_variables : input_values} and it is used when we have 
#input variables in our prompt. but when we don't have input variables in our prompt
# then we can use template.format() or template.invoke("some random text") and it will return the prompt in string format.



# print(prompt)

# response = model.invoke(prompt)
# # print(response.content)

# final_response = parser.invoke(response)
# print(final_response)






##################################################################################################

# using chains

chain = template | model | parser

final_response = chain.invoke({})
print(final_response)