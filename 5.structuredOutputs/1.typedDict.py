from langchain_core.output_parsers import string
from typing import TypedDict

class Person(TypedDict):
    name : str
    age : int
    gender : str

new_person : Person = {'name':'Sajal', 'age':21, 'gender':'Male'}

print(new_person)

