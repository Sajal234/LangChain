from sympy import sturm
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


# pydantic is more powerful than TypedDict and helps in data validation and data structuring with better type checking capabilities and support for nested structures


class Student(BaseModel):
    name : str = 'Nitish' #default value is Nitish
    age : Optional[int] = None #default value is None and its optional
    email : EmailStr # will validate the email
    cgpa : float = Field(gt=0, lt=10, default=5, description='Average cgpa of student on the scale of 10')


new_student = {'name':'Sajal'}
# pydantic will convert this string 23 into number 23 only if int() can do this conversion and hence it will return proper object, if it cannot do this conversion then it will raise an error.
new_student = {'age':'23', 'email':'abc@gmail.com', 'cgpa':8.1}



# this doesnt work because 32 is not a string and type hinting will prevent it from getting assigned.
# new_student = {'name':32}

student = Student(**new_student)
student_dict = dict(student)

print(student)
print(type(student))

print(student_dict['age'])
    
student_json = student.model_dump_json()
print(student_json)
print(type(student_json))
