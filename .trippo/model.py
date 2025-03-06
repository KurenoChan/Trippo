# import os
# print("Current working directory:", os.getcwd())

# import json
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import LabelEncoder
# import nltk

# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

# # Load intents.json
# with open("trippo/intents.json") as file:
#     data = json.load(file)

# # Prepare training data
# lemmatizer = WordNetLemmatizer()
# words = []
# classes = []
# documents = []
# ignore_words = ["?", "!", ".", ","]

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         word_list = word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent["tag"]))
#     if intent["tag"] not in classes:
#         classes.append(intent["tag"])

# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(set(words))
# classes = sorted(set(classes))

# # Create training data
# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
#     for w in words:
#         bag.append(1) if w in word_patterns else bag.append(0)
    
#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])

# random.shuffle(training)
# training = np.array(training, dtype=object)
# train_x = np.array([i[0] for i in training])
# train_y = np.array([i[1] for i in training])

# # Build the model
# model = Sequential([
#     Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
#     Dropout(0.5),
#     Dense(64, activation="relu"),
#     Dropout(0.5),
#     Dense(len(classes), activation="softmax")
# ])

# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# # Train the model
# model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# # Save the model
# model.save("chatbot_model.h5")
# np.save("words.npy", words)
# np.save("classes.npy", classes)
