from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

#schemas
class ThemeItem(TypedDict):
    theme: Annotated[str, "The aspect or theme of the movie, e.g. Acting, Cinematography"]
    sentiment: Annotated[Literal["positive", "negative"], "Sentiment for this theme: positive or negative"]
    reasons : Annotated[list[str], "Reasons for this sentiment"]

class Review(TypedDict):
    key_themes: Annotated[list[ThemeItem], "List of key themes with their individual sentiments"]
    summary: Annotated[str, "Brief summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Overall sentiment of the review"]
    rating: Annotated[int, "Rating of the review from 1 to 5, don't overrate it, just be natural"]
    recommendation : Annotated[Optional[str], "Recommendation for this movie, recommend or not"]


structured_model = model.with_structured_output(Review)

response = structured_model.invoke("""
Drishyam 3 is the best movie I have ever seen. It's a masterpiece. 
Acting of Jaideep Ahlawat is just awesome!
Part 1 and 2 were also good. But this one is just mind blowing.
Apart from the storyline the cinematography and direction is slightly disappointing.
They can do better and acting wise except Jaideep and Ajay Devgan, rest of the cast is not upto the mark.
So, i would recommend everyone to watch this movie. But don't expect it to be as good as part 1 and 2.
""")

print(f"\nSummary : {response['summary']}\n")
print(f"Overall Sentiment : {response['sentiment']}\n")
print(f"Overall Rating : {response['rating']}\n")

print("="*50)
print("\nKey Themes :\n")
for item in response['key_themes']:
    print(f"Theme : {item['theme']}")
    print(f"Sentiment : {item['sentiment']}")
    for reason in item['reasons']:
        print(f"  - {reason}")
    print("-"*30)
    
print(f"\nRecommendation : {response['recommendation']}")
