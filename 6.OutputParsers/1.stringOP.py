from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n{text}",
    input_variables=["text"]
)


######################################################################################################################

# normal chain without any parser

# prompt1 = template1.invoke({'topic' : 'Black Hole'})

# response_report = model.invoke(prompt1)

# prompt2 = template2.invoke({'text' : response_report.content})

# response_summary = model.invoke(prompt2)

# print("\n" + "="*50 + "\n")
# print("Report : \n\n", response_report.content)
# print("\n" + "="*50 + "\n")
# print("\nSummary : \n\n", response_summary.content)




############################################################################################################

# using string output parser
parser = StrOutputParser()
chain = template1 | model | parser | template2 |  model | parser

result = chain.invoke({'topic' : 'Black Hole'})

print("\n" + "="*50 + "\n")
print("Summary : \n\n", result) 

