from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel, Field
 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

#schemas
class ThemeItem(BaseModel):
    theme: str = Field(description="The aspect or theme of the movie, e.g. Acting, Cinematography")
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment for this theme: positive or negative")
    reasons : list[str] = Field(description="Reasons for this sentiment")

class Review(BaseModel):
    key_themes: list[ThemeItem] = Field(description="List of key themes with their individual sentiments")
    summary: str = Field(description="Brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the review")
    rating: int = Field(description="Rating of the review from 1 to 5, don't overrate it, just be natural")
    recommendation : Optional[str] = Field(description="Recommendation for this movie, recommend or not")
    name: Optional[str] = Field(description="Write the name of the reviewer if given, if not then leave it as empty string", default="")


structured_model = model.with_structured_output(Review)

response = structured_model.invoke("""
Drishyam 3 is the best movie I have ever seen. It's a masterpiece. 
Acting of Jaideep Ahlawat is just awesome!
Part 1 and 2 were also good. But this one is just mind blowing.
Apart from the storyline the cinematography and direction is slightly disappointing.
They can do better and acting wise except Jaideep and Ajay Devgan, rest of the cast is not upto the mark.
So, i would recommend everyone to watch this movie. But don't expect it to be as good as part 1 and 2.

- Sajal
""")

print(f"\nName : {response.name}")
print(f"\nSummary : {response.summary}\n")
print(f"Overall Sentiment : {response.sentiment}\n")
print(f"Overall Rating : {response.rating}\n")

print("="*50)
print("\nKey Themes :\n")
for item in response.key_themes:
    print(f"Theme : {item.theme}")
    print(f"Sentiment : {item.sentiment}")
    for reason in item.reasons:
        print(f"  - {reason}")
    print("-"*30)
    
print(f"\nRecommendation : {response.recommendation}")

