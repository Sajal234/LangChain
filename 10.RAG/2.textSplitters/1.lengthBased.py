from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



# text = """Minecraft is a sandbox video game developed by Mojang Studios and created by Markus Persson. Released publicly in 2009 and officially launched in November 2011, it has become the best-selling video game of all time.

# In Minecraft, players explore a blocky, procedurally generated 3D world with infinite terrain. They discover and extract raw materials, craft tools and items, and build structures or earthworks. Depending on the chosen game mode, players can fight computer-controlled hostile mobs, as well as cooperate with or compete against other players.

# The two main game modes are Survival mode and Creative mode. In Survival mode, players must acquire resources such as wood and stone to build the world and maintain health. They must also manage hunger and protect themselves from nocturnal monsters like zombies, skeletons, and creepers. In Creative mode, players have access to unlimited resources and the ability to fly, allowing them to focus entirely on building large and imaginative architectural projects without facing threats.

# The game also features alternative dimensions. The Nether is a dangerous, hell-like dimension filled with lava, fortresses, and unique resources like quartz and blaze rods. Beyond the Nether lies The End, a barren realm made of end stone where players can face the Ender Dragon, the primary boss of the game. Defeating the Ender Dragon displays the game's credits and opens portals to outer End islands where Elytra wings and shulker boxes can be found.

# Redstone is another core feature that functions like electrical circuitry within the game. Using redstone dust, repeaters, pistons, and comparators, players can build complex mechanical devices ranging from automated sugarcane farms and piston doors to fully functioning in-game computers."""


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
# result = splitter.split_text(text)

# print(len(result))
# print(type(result))
# print(result)
# print(len(result[0]))
# print(f'''\n\n{result[0]}''')




loader = PyPDFLoader('10.RAG/1.DocumentLoaders/intro.pdf')

docs = loader.load()
print(len(docs))

result = splitter.split_documents(docs)

print(len(result))
print(result)
print(f'''\n\n{result[0]}''')
