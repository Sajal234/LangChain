from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel, Field
 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

#schemas
json_schema = {
    "title": "Review",
    "description": "Schema for movie review analysis",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the reviewer if mentioned, otherwise an empty string"
        },
        "summary": {
            "type": "string",
            "description": "Brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the review"
        },
        "rating": {
            "type": "integer",
            "description": "Rating of the movie from 1 to 10, don't overrate it, just be natural"
        },
        "key_themes": {
            "type": "array",
            "description": "List of key themes with their individual sentiments and reasons",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "Aspect or theme of the movie, e.g., Acting, Cinematography"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative"],
                        "description": "Sentiment for this specific theme"
                    },
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reasons or evidence from the review supporting this sentiment"
                    }
                },
                "required": ["theme", "sentiment", "reasons"]
            }
        },
        "recommendation": {
            "type": "string",
            "description": "Recommendation whether to watch the movie or not"
        }
    },
    "required": ["summary", "sentiment", "rating", "key_themes"]
}


structured_model = model.with_structured_output(json_schema)

response = structured_model.invoke("""
Drishyam 3 is the best movie I have ever seen. It's a masterpiece. 
Acting of Jaideep Ahlawat is just awesome!
Part 1 and 2 were also good. But this one is just mind blowing.
Apart from the storyline the cinematography and direction is slightly disappointing.
They can do better and acting wise except Jaideep and Ajay Devgan, rest of the cast is not upto the mark.
So, i would recommend everyone to watch this movie. But don't expect it to be as good as part 1 and 2.

- Sajal
""")


print(response)
# print(f"\nName : {response['name']}")
# print(f"\nSummary : {response['summary']}\n")
# print(f"Overall Sentiment : {response['sentiment']}\n")
# print(f"Overall Rating : {response['rating']}\n")

# print("="*50)
# print("\nKey Themes :\n")
# for item in response['key_themes']:
#     print(f"Theme : {item['theme']}")
#     print(f"Sentiment : {item['sentiment']}")
#     for reason in item['reasons']:
#         print(f"  - {reason}")
#     print("-"*30)
    
# print(f"\nRecommendation : {response['recommendation']}")

