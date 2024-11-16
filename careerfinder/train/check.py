import pickle

# Load the saved model and vectorizer
with open('model_and_vectorizer.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

def predict_intent(user_input):
    # Transform the user input using the vectorizer
    user_input_vectorized = vectorizer.transform([user_input])
    
    # Get the predicted intent
    predicted_intent = model.predict(user_input_vectorized)
    
    # Return the predicted intent
    return predicted_intent[0]

# Example usage
user_input = "hi"
predicted_intent = predict_intent(user_input)

print(f"The predicted intent is: {predicted_intent}")

def get_response(intent):
    responses = {
        "Greeting": "Hello! How can I assist you today?",
        "Career_Guidance": "I can help you with career guidance. What field are you interested in?",
        "Science_Streams": "Science streams include Physics, Chemistry, Biology, and more.",
        # Add other intents and their corresponding responses
    }
    return responses.get(intent, "I'm sorry, I didn't understand that.")

# Example interaction
user_input = "I want to know more about career options in science."
predicted_intent = predict_intent(user_input)
response = get_response(predicted_intent)
print(response)
