from torch.fx.passes import splitter_base
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """import math

class Calculator:
    \"\"\"A simple calculator class for arithmetic operations.\"\"\"
    def __init__(self, name="BasicCalc"):
        self.name = name

    def add(self, a: float, b: float) -> float:
        \"\"\"Return the sum of two numbers.\"\"\"
        return a + b

    def subtract(self, a: float, b: float) -> float:
        \"\"\"Return the difference of two numbers.\"\"\"
        return a - b

    def multiply(self, a: float, b: float) -> float:
        \"\"\"Return the product of two numbers.\"\"\"
        return a * b

    def divide(self, a: float, b: float) -> float:
        \"\"\"Return quotient, raises error if division by zero.\"\"\"
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b


class AdvancedCalculator(Calculator):
    \"\"\"Advanced calculator supporting power and square root operations.\"\"\"
    def power(self, base: float, exp: float) -> float:
        return math.pow(base, exp)

    def square_root(self, value: float) -> float:
        if value < 0:
            raise ValueError("Cannot take square root of negative number.")
        return math.sqrt(value)


def process_numbers(numbers: list[float]) -> dict:
    \"\"\"Utility function to summarize a list of numbers.\"\"\"
    calc = Calculator()
    total = 0
    for num in numbers:
        total = calc.add(total, num)

    average = calc.divide(total, len(numbers)) if numbers else 0
    return {
        "count": len(numbers),
        "total": total,
        "average": average
    }


if __name__ == "__main__":
    nums = [10.0, 20.0, 30.0, 40.0]
    result = process_numbers(nums)
    print("Summary:", result)
"""



# we can do same with different language by using from_language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)


chunks = splitter.split_text(python_code)

print(len(chunks))
print(f'\n\n{chunks[10]}')