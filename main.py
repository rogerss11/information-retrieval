from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

app = FastAPI()

CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
if not CAMPUSAI_API_KEY:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://campusai.compute.dtu.dk/api"
)

history = []

class TextInput(BaseModel):
    text: str

def campusai_extract_persons(text):
    prompt = f"""Extract all the person names from the text provided below.:
    Return the names as a list of names separated by #.
    Example: <name1>#<name2>#...
    Return only the list. No additional text.
    If no names are found, return an empty list. Text: {text}"""

    history.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gemma3:latest",
        messages=history,
        stream=False  # explicitly disable streaming
    )
    reply = response.choices[0].message.content.split("#")
    print(reply)
    filtered_reply = [x for x in reply if x]
    return filtered_reply

@app.post("/v1/extract-persons")
def extract_persons(text: TextInput):
    reply = campusai_extract_persons(text.text)
    return {"persons": reply}

if __name__ == "__main__":
    print("Enter ’exit’ to end the session.")
    while True:
        # Prompt the user
        user_input = input(" You : ")
        if user_input.lower() == "exit":
            break
        history.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model="gemma3:latest",
            messages=history,
            stream=False  # explicitly disable streaming
        )
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}" + "\n" + "-" * 40)
        history.append({"role": "assistant", "content": reply})