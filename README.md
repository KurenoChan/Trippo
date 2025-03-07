Trippo - AI Travel Assistant Chatbot

Trippo is an AI-powered travel assistant chatbot designed to provide intelligent responses using NLP. This project integrates machine learning with Discord to create a conversational bot.

Project Structure










.venv\Scripts\Activate

python trippo/trippo_bot.py

python trippo/main.py
python trippo/model.py
python trippo/utils.py


SAMPLE CHATBOT:
https://www.youtube.com/watch?v=9KZwRBg4-P0


1. intents.json (Defines intents for NLP)
This file will store predefined user intents (greetings, questions, etc.).

2. model.py (Handles NLP training using TensorFlow/Keras) [MACHINE_LEARNING]
This script will preprocess text, train a simple chatbot model, and save it.

3. utils.py (Handles NLP response prediction) [RESPONSE_PREDICTIONS]
This script will load the trained model and process user input.

4. trippo_bot.py (Integrates NLP into the Discord bot)
This script allows the bot to process user commands and use NLP for responses.



--------------------------------------------------------------
STEPS:
Run model.py to train and save chatbot_model.h5
After the model is trained and saved, run trippo_bot.py again:
--------------------------------------------------------------



How Everything Works Together
Training Phase (model.py)

Loads intents.json, tokenizes & lemmatizes text.
Creates training data and trains a neural network.
Saves the trained model (chatbot_model.h5), words, and classes.
NLP Prediction (utils.py)

Loads the trained model and necessary files.
Processes new user input and predicts the most relevant response.
Discord Bot (trippo_bot.py)

Listens for messages and commands.
Calls predict_response() from utils.py to generate replies.