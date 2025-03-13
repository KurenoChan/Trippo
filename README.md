# Trippo - AI Travel Assistant Chatbot

Trippo is an AI-powered travel assistant chatbot designed to provide intelligent responses using NLP. This project integrates machine learning with Discord to create a conversational bot.

## Table of Contents

1.0 [Project Structure]([#1.0-project-structure](https://github.com/KurenoChan/Trippo/blob/main/README.md#10-project-structure))<br/>
2.0 [Setup & Installation](#2.0-setup-installatiob)<br/>
    &emsp;&emsp;2.1 [Activate Virtual Environment](#2.1-activate-virtual-environment)<br/>
    &emsp;&emsp;2.2 [Install Dependencies](#2.2-install-dependencies)<br/>
    &emsp;&emsp;2.3 [Train the Model](#2.3-train-the-model)<br/>
    &emsp;&emsp;2.4 [Start the Discord Bot](#2.4-start-the-discord-bot)<br/>
3.0 [How It Works](#3.0-how-it-works)<br/>
4.0 [Example Chatbot Demo](#4.0-example-chatbot-demo)

---------------------
## 1.0 Project Structure

```
.trippo/
│── intents.json        # Defines chatbot intents (greetings, questions, etc.)
│── model.py            # Trains and saves the NLP model (TensorFlow/Keras)
│── utils.py            # Handles NLP response predictions
│── trippo_bot.py       # Discord bot integration
│── chatbot_model.h5    # Saved trained model (generated after training)
```

-----------------------------
## 2.0 Setup & Installation

### 2.1 Activate Virtual Environment
`.venv\Scripts\Activate`

### 2.2 Install Dependencies
`pip install -r requirements.txt`

### 2.3 Train the Model
Run the following command to train the chatbot model and generate chatbot_model.h5:
`python .trippo/model.py`

### 2.4 Start the Discord Bot
After training is complete, start the chatbot:

`python .trippo/trippo_bot.py`

------------
## 3.0 How It Works

### 1. Training Phase (model.py)
- Loads `intents.json`, tokenizes, and lemmatizes text.
- Prepares training data and trains a neural network using TensorFlow/Keras.
- Saves the trained model (`chatbot_model.h5`), words, and class mappings.

### 2. NLP Prediction (utils.py)
- Loads the trained model and required files.
- Processes user input and predicts the best response.

### 3. Discord Bot (trippo_bot.py)
- Listens for user messages and commands.
- Calls `predict_response()` from `utils.py` to generate a response.
Example Chatbot Demo

> Check out this [sample chatbot implementation](https://www.youtube.com/watch?v=9KZwRBg4-P0) for reference.
