from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('key.env')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found in key.env file")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def get_response(user_message):
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::Bdw2s1cV", 
            messages=[
                {"role": "system", "content": "You are a helpful electronics store assistant for ElectroStore. You help customers with product information, technical support, and general electronics knowledge."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

print("Welcome to ElectroStore Assistant!")
print("Type 'quit' to exit")

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
        
    response = get_response(user_input)
    print("\nAssistant:", response)
