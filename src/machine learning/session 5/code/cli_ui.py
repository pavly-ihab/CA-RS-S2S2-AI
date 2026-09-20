
from model import get_response

def chatbot():
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ").lower()
        response = get_response(user_input)
        print("Chatbot", response)
        
        if user_input == "goodbye":
            break
        