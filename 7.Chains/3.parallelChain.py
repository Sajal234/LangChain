from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1 = PromptTemplate(
    template='Generate short and simple notes for the text {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate five short question answers for the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n Notes: \n {notes} \n Quiz: \n {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model1 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Minecraft is a sandbox video game developed by Mojang Studios. The game was created by Markus "Notch" Persson in 2009 and officially released in 2011. Minecraft allows players to explore a blocky, procedurally generated 3D world with infinite terrain, discover and extract raw materials, craft tools and items, and build structures or earthworks. Depending on the game mode, players can fight computer-controlled mobs, as well as cooperate with or compete against other players in the same world. The game's main modes are Survival, where players must acquire resources to survive, and Creative, where players have unlimited resources and the ability to fly. Minecraft is available on multiple platforms, including PC, consoles, and mobile devices. It has sold over 300 million copies worldwide, making it the best-selling video game of all time.
"""

response = chain.invoke({'topic' : text, 'text' : text})

print(response)