# import json
# import numpy as np
# import tensorflow as tf
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Load necessary files
# lemmatizer = WordNetLemmatizer()
# model = tf.keras.models.load_model("chatbot_model.h5")
# words = np.load("words.npy", allow_pickle=True)
# classes = np.load("classes.npy", allow_pickle=True)

# # Load intents.json
# with open("trippo/intents.json") as file:
#     data = json.load(file)

# # Function to preprocess user input
# def preprocess_text(sentence):
#     tokens = word_tokenize(sentence)
#     tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
#     bag = [1 if w in tokens else 0 for w in words]
#     return np.array([bag])

# # Function to predict response
# def predict_response(text):
#     processed_input = preprocess_text(text)
#     predictions = model.predict(processed_input)[0]
#     max_index = np.argmax(predictions)
#     tag = classes[max_index]

#     for intent in data["intents"]:
#         if intent["tag"] == tag:
#             return np.random.choice(intent["responses"])
    
#     return "I'm not sure how to respond to that."
