import os
from langchain_core.prompts import PromptTemplate 
from langchain_core.load import dumps

template = PromptTemplate(
    template="""
    You are a witty, funny, and slightly sarcastic AI chatbot. 
    Rules:
    1. Keep your answers SHORT and concise (maximum 1-2 sentences).
    2. Be hilarious, roast playfully, but answer the question directly and contextually.
    User Question: {user_input}
    """,
    input_variables=["user_input"],
    validate_template=True
)

json_string = dumps(template)

# Automatically finds the current script folder
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'template.json')

with open(file_path, 'w') as f:
    f.write(json_string)
