import os

from dotenv import load_dotenv


load_dotenv('/Users/ajitk/Desktop/GenAI/key.env')


gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

print("Gemini Key:", gemini_api_key)
print("Groq Key:", groq_api_key)
from groq import Groq
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown


def call_gemini(prompt,system_prompt=None):
    
    def to_markdown(text):
        text = text.replace('.', ' *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    

    genai.configure(api_key=gemini_api_key)
    
    model = genai.GenerativeModel('gemini-1.5-flash')

    response = model.generate_content(prompt)
    
    return response.text.strip()
    

def call_groq(prompt):
    client = Groq(
            api_key = groq_api_key,
)
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama3-70b-8192",
)

    return chat_completion.choices[0].message.content

def run_chat_agent():
    print("Welcome to Groq Chatting agent! Type 'exit' or 'quit' to end the conversation")
    while 1:
        user = input("You: ").strip()
        if user.lower() == "exit" or user.lower() == "quit":
            print("Goodbye!")
            break
        elif user.lower == "":
            print("Please enter a non-empty message.")
            continue

        else:
            response = call_groq(user)
            
        print("GROQ_AGENT: ",response)
        print("You: Is it correct?")
        response2 = call_gemini(response,system_prompt=None)
        print("GEMINI_CRTIC: ",response2)

if __name__ == "__main__":
    run_chat_agent()
        


            


