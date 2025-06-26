# Trippo - AI Travel Assistant Chatbot

Trippo is a tourism-focused AI assistant that helps users with travel-related inquiries. Powered by custom-trained machine learning models and multilingual capabilities, it can answer questions about tourist attractions, accommodations, transportation, traditional food, culture, and more, supporting both structured and unstructured datasets.

# Table of Contents


1.0 [Project Structure](#10-project-structure)<br/>
2.0 [Setup & Installation](#20-setup--installation)<br/>
  2.1 [Setting Up the Bot](#21-setting-up-the-bot)<br/>
    2.1.1 [Create Virtual Environment](#211-create-virtual-environment)<br/>
    2.1.2 [Install Packages](#212-install-packages)<br/>
    2.1.3 [Data Preprocessing](#213-data-preprocessing)<br/>
      2.1.3.1 [Choose the Appropriate Processor](#2131-choose-the-appropriate-processor)<br/>
      2.1.3.2 [Export Preprocessed File](#2132-export-preprocessed-file)<br/>
    2.1.4 [Train the Model](#214-train-the-model)<br/>
  2.2 [Running the Bot](#22-running-the-bot)<br/>
    2.2.1 [Activate the Virtual Environment](#221-activate-the-virtual-environment)<br/>
    2.2.2 [Run the Bot](#222-run-the-bot)<br/>
3.0 [How It Works](#30-how-it-works)<br/>
4.0 [Example Chatbot Demo](#40-example-chatbot-demo)

---

# 1.0 Project Structure

```
TrippoBot/
│── data/
│   ├── processed/        # Processed data (JSON format)
│   └── raw/              # Raw input datasets/materials
│
│── model/
│   ├── file_processor/   # Tools to process different file types
│   │   ├── csv_processor.ipynb
│   │   ├── excel_processor.ipynb
│   │   ├── pdf_processor.ipynb
│   │   ├── txt_processor.ipynb
│   │   └── web_processor.ipynb
│   │
│   ├── language_manager/ # Language tools (detection, translation, etc.)
│   │   ├── lang_detector.py
│   │   ├── lang_reservedWord.py
│   │   ├── lang_spelling.py
│   │   └── lang_translator.py
│   │
│   └── model_manager/
│       ├── model_llm.ipynb
│       ├── model_trainer.ipynb   # Train and export models
│       └── models.pkl            # Trained models
│
│── trippo/
│   └── trippo.py          # Main bot script
│
│── util/
│   └── run_ipynb.py       # Utility to run notebooks programmatically
│
│── requirements.ipynb     # Dependencies and installation commands
│── README.md              # Project documentation
```

---

# 2.0 Setup & Installation

## 2.1 Setting Up the Bot

### 2.1.1 Create Virtual Environment

`python -m venv .venv`

### 2.1.2 Install Packages

Activate the virtual environment.

```
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On Mac/Linux
```

Then, open and run `requirements.ipynb` to install **ALL** necessary dependencies.

### 2.1.3 Data Preprocessing
#### 2.1.3.1 Choose the Appropriate Processor
1) Navigate to `model/file_processor/`.
   
2) Select the processor based on your input file format:
    > CSV     : `csv_processor.ipynb`
    > Excel   : `excel_processor.ipynb`
    > PDF     : `pdf_processor.ipynb`
    > TXT     : `txt_processor.ipynb`
    > Website : `web_processor.ipynb`

#### 2.1.3.2 Export Preprocessed File
Export the cleaned result into the `data/processed/` folder in **JSON** format.

### 2.1.4 Train the Model
1) Open `model/model_manager/model_trainer.ipynb`
2) Train the model using the processed data.
3) Export the model(s) and load into `models.pkl` under the same directory.

## 2.2 Running the Bot
### 2.2.1 Activate the Virtual Environment
```
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On Mac/Linux
```

### 2.2.2 Run the Bot
```python trippo/trippo.py```

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
