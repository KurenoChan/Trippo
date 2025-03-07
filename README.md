# Trippo - AI Travel Assistant Chatbot

Trippo is an AI-powered travel assistant chatbot designed to provide intelligent responses using NLP. This project integrates machine learning with Discord to create a conversational bot.

## Table of Contents

    1.0 [Project Structure](#1.0-project-structure)<br/>
    2.0 [Setup & Installation](#2.0-setup-installatiob)<br/>
        2.1 Activate Virtual Environment
        2.2 Install Dependencies
        2.3 Train the Model
        2.4 Start the Discord Bot
3.0 How It Works
4.0 Example Chatbot Demo

---------------------
### 1.0 Project Structure

```
trippo/
│── intents.json        # Defines chatbot intents (greetings, questions, etc.)
│── model.py            # Trains and saves the NLP model (TensorFlow/Keras)
│── utils.py            # Handles NLP response predictions
│── trippo_bot.py       # Discord bot integration
│── chatbot_model.h5    # Saved trained model (generated after training)
```

### 2.0 Setup & Installation

1. Activate Virtual Environment
.venv\Scripts\Activate
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model
Run the following command to train the chatbot model and generate chatbot_model.h5:

python trippo/model.py
4. Start the Discord Bot
After training is complete, start the chatbot:

python trippo/trippo_bot.py
How It Works

1. Training Phase (model.py)
- Loads `intents.json`, tokenizes, and lemmatizes text.
- Prepares training data and trains a neural network using TensorFlow/Keras.
- Saves the trained model (`chatbot_model.h5`), words, and class mappings.
2. NLP Prediction (utils.py)
- Loads the trained model and required files.
- Processes user input and predicts the best response.
3. Discord Bot (trippo_bot.py)
- Listens for user messages and commands.
- Calls `predict_response()` from `utils.py` to generate a response.
Example Chatbot Demo

Check out this sample chatbot implementation for reference.







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