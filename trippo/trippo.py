import os
import random
import discord
import joblib
import numpy as np
import pandas as pd
from discord.ext import commands
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import sys
from openai import OpenAI

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import your language modules
from model.language_manager.lang_translator import translate_text
from model.language_manager.lang_spelling import correct_spelling
from model.language_manager.lang_reservedWord import is_reserved_word, is_slang_word, SLANG_WORDS, APP_WORDS
from model.language_manager.lang_detector import detect_language

# Load environment variables
load_dotenv()

# Set Discord intents
intents = discord.Intents.default()
intents.message_content = True

# Create bot
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Language Configuration
DEFAULT_LANGUAGE = 'en'
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh-cn', 'ja']

class LanguageManager:
    def __init__(self):
        self.target_language = DEFAULT_LANGUAGE
        self.reserved_words = APP_WORDS
        self.slang_words = SLANG_WORDS

    def is_reserved_word(self, word):
        return is_reserved_word(word)
    
    def is_slang_word(self, word):
        return is_slang_word(word)

    def get_slang_replacement(self, word):
        return self.slang_words.get(word.lower(), word)

    @lru_cache(maxsize=1000)
    def translate_text(self, text, target_language):
        return translate_text(text, target_language)

    def detect_language(self, text):
        return detect_language(text)

    def correct_spelling(self, text):
        return correct_spelling(text)

class TravelModelAssistant:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "../model/training_manager/models.pkl")
        self.dataset_path = os.path.join(script_dir, "../data/processed/processed_destinations.json")
        self.tag_examples_path = os.path.join(script_dir, "../data/processed/processed_tourism_data.json")
        
        self.models_path = model_path
        self.trained_models = []
        self.embedder = None
        self.model_descriptions = {}
        self.description_embeddings = None
        self.language_manager = LanguageManager()
        self.target_language = DEFAULT_LANGUAGE
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gpt_enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.destination_data = {}
        self.tag_examples = {}
        
        print(f"Looking for models at: {model_path}")
        self.load_models()
        self.load_destination_data()
        self.load_tag_examples()
        self.initialize_embedder()

    def load_models(self):
        try:
            if os.path.exists(self.models_path):
                print("Found models file, attempting to load...")
                self.trained_models = joblib.load(self.models_path)
                print(f"✅ Loaded {len(self.trained_models)} trained models")
                
                # Build model descriptions
                self.model_descriptions = {}
                for idx, model_info in enumerate(self.trained_models):
                    if isinstance(model_info, dict):
                        predictor = model_info.get('predictor', 'Unknown')
                        features = ', '.join(model_info.get('features', []))
                        desc = f"Model {idx+1}: Predicts {predictor} using {features}"
                        self.model_descriptions[f"model_{idx+1}"] = desc
                    else:
                        self.model_descriptions[f"model_{idx+1}"] = f"Model {idx+1}"
            else:
                print(f"❌ Models file not found at {self.models_path}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.trained_models = []

    def load_destination_data(self):
        try:
            import json
            if os.path.exists(self.dataset_path):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.destination_data = json.load(f)
                print(f"✅ Loaded destination data for {len(self.destination_data)} locations")
            else:
                print(f"❌ Destination data file not found at {self.dataset_path}")
        except Exception as e:
            print(f"❌ Error loading destination data: {e}")
            self.destination_data = {}

    def load_tag_examples(self):
        try:
            import json
            if os.path.exists(self.tag_examples_path):
                with open(self.tag_examples_path, 'r', encoding='utf-8') as f:
                    self.tag_examples = json.load(f)
                print(f"✅ Loaded {len(self.tag_examples)} tag examples")
            else:
                print(f"❌ Tag examples file not found at {self.tag_examples_path}")
        except Exception as e:
            print(f"❌ Error loading tag examples: {e}")
            self.tag_examples = {}

    def initialize_embedder(self):
        try:
            self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("✅ Loaded sentence embedding model")
            if self.trained_models and self.model_descriptions:
                texts = list(self.model_descriptions.values())
                self.description_embeddings = self.embedder.encode(texts)
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.embedder = None

    async def get_gpt_fallback(self, user_input):
        """Get a fallback response from GPT-3.5 when our models can't answer"""
        if not self.gpt_enabled:
            return None
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful travel assistant. Provide concise, accurate answers to travel-related questions."
                    },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT-3.5 fallback error: {e}")
            return None

    def find_destination_in_data(self, destination_name):
        """Search for a destination in our dataset"""
        if not self.destination_data:
            return None
            
        destination_name = destination_name.lower().strip()
        for dest in self.destination_data:
            if dest['Destination'].lower() == destination_name:
                return dest
        return None

    def select_model(self, user_input):
        """Select the most appropriate model based on user input"""
        if not self.trained_models:
            return None
            
        user_input_lower = user_input.lower()
        
        # First try to match text classification models (for tag prediction)
        for model in self.trained_models:
            if model.get('vectorizer') and 'sentence' in model.get('features', []):
                return model
        
        # Then try to match based on predictor
        for model in self.trained_models:
            predictor = model.get('predictor', '').lower()
            if predictor in user_input_lower:
                return model
        
        # Then try to match based on features
        for model in self.trained_models:
            features = model.get('features', [])
            if any(feat.lower() in user_input_lower for feat in features):
                return model
                
        # Finally try semantic similarity
        if self.embedder and self.description_embeddings is not None:
            try:
                user_embedding = self.embedder.encode([user_input])
                sims = cosine_similarity(user_embedding, self.description_embeddings)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] > 0.4:  # Only use if similarity is above threshold
                    return self.trained_models[best_idx]
            except Exception as e:
                print(f"Model selection error: {e}")
        
        return None

    def prepare_input(self, model_info, user_input):
        """Prepare input data based on model type"""
        if model_info.get('vectorizer'):
            # Text classification model
            return model_info['vectorizer'].transform([user_input])
        else:
            # Structured data model
            features = {}
            input_text = user_input.lower()
            
            # Handle one-hot encoded features
            expected_columns = model_info.get('X_train_columns', [])
            for col in expected_columns:
                if '_' in col:  # One-hot encoded feature
                    feature, value = col.split('_', 1)
                    if feature.lower() in input_text and value.lower() in input_text:
                        features[col] = 1
                    else:
                        features[col] = 0
                else:  # Regular feature
                    features[col] = 1 if col.lower() in input_text else 0
            
            # Convert to array in correct column order and reshape for sklearn
            return np.array([features.get(col, 0) for col in expected_columns]).reshape(1, -1)

    def generate_destination_response(self, destination_info):
        if not destination_info:
            return None
            
        # Create embed-like formatting for Discord
        response = [
            f"**✈️ {destination_info['Destination']} Overview**",
            f"**Country:** {destination_info.get('Country', 'Unknown')}"
        ]
        
        if 'Famous Foods' in destination_info:
            foods = ', '.join(destination_info['Famous Foods']) if isinstance(destination_info['Famous Foods'], list) else destination_info['Famous Foods']
            response.append(f"**Famous Foods:** {foods}")
        
        if 'Majority Religion' in destination_info:
            response.append(f"**Religion:** {destination_info['Majority Religion']}")
        
        if 'Best Time to Visit' in destination_info:
            response.append(f"**Best Time to Visit:** {destination_info['Best Time to Visit']}")
        
        if 'Description' in destination_info:
            response.append(f"\n*{destination_info['Description']}*")
        
        return '\n'.join(response)

    def generate_response(self, model_info, user_input):
        try:
            model = model_info.get("model")
            if model is None:
                return None
                
            # First try to find exact destination match
            destination_name = self.extract_destination_name(user_input)
            if destination_name:
                destination_info = self.find_destination_in_data(destination_name)
                if destination_info:
                    response = self.generate_destination_response(destination_info)
                    if response:
                        return f"{response}\n(Confidence: 95% - from verified dataset)"
            
            # If no exact match, use the model
            processed_input = self.prepare_input(model_info, user_input)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            label_encoder = model_info.get("label_encoder")
            if label_encoder:
                prediction = label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence score
            confidence = 0.8  # Default confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed_input)
                confidence = np.max(proba)
            elif hasattr(model, "kneighbors"):  # For KNN models
                distances, _ = model.kneighbors(processed_input)
                confidence = 1 - (np.mean(distances) / 10)  # Simple confidence based on distance
            
            # Format response
            predictor = model_info.get('predictor', 'information')
            response = [f"✈️ My analysis shows with {confidence:.0%} confidence that the {predictor} is: **{prediction}**"]
            
            # Add example sentences if this is a tag prediction
            if model_info.get('vectorizer') and 'sentence' in model_info.get('features', []):
                examples = self.get_tag_examples(prediction)
                if examples:
                    response.append("\n**Example sentences with this tag:**")
                    response.extend(f"- \"{ex}\"" for ex in examples[:3])  # Show max 3 examples
            
            return '\n'.join(response)
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return None

    def get_tag_examples(self, tag):
        """Get example sentences for a given tag"""
        if not self.tag_examples:
            return []
        
        # Find examples matching the tag (case insensitive)
        examples = [ex['sentence'] for ex in self.tag_examples 
                   if ex['tag'].lower() == tag.lower()]
        
        # If no exact matches, try partial matches
        if not examples:
            examples = [ex['sentence'] for ex in self.tag_examples 
                       if tag.lower() in ex['tag'].lower()]
        
        return examples[:5]  # Return up to 5 examples

    def extract_destination_name(self, user_input):
        """Extract destination name from user input"""
        # Simple extraction - look for words that match our destination names
        if not self.destination_data:
            return None
            
        user_input_lower = user_input.lower()
        for dest in self.destination_data:
            dest_name = dest['Destination'].lower()
            if dest_name in user_input_lower:
                return dest['Destination']
        return None

    async def get_travel_response(self, user_input):
        if not self.trained_models:
            return self.language_manager.translate_text(
                "⚠️ My travel knowledge isn't loaded yet. Please try again later.",
                self.target_language
            )
        
        processed_input = self.language_manager.correct_spelling(user_input)
        input_lower = processed_input.lower()
        
        # Handle tag prediction queries (both formats)
        if 'tag of' in input_lower or 'tag for' in input_lower:
            sentence = processed_input.split('of', 1)[1].strip() if 'of' in input_lower else processed_input.split('for', 1)[1].strip()
            return await self.predict_tag(sentence)
        
        # Try to extract destination name from input
        destination_name = self.extract_destination_name(processed_input)
        
        # If we found a destination name, try to answer from our dataset first
        if destination_name:
            destination_info = self.find_destination_in_data(destination_name)
            if destination_info:
                # Food-related questions
                if any(keyword in input_lower for keyword in ['food', 'cuisine', 'dish', 'eat', 'meal', 'restaurant']):
                    if 'Famous Foods' in destination_info:
                        foods = destination_info['Famous Foods']
                        if isinstance(foods, list):
                            foods = ', '.join(foods)
                        response = f"🍽️ The famous foods in {destination_name} include: {foods} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # Category questions
                elif any(keyword in input_lower for keyword in ['category', 'type', 'kind', 'classification']):
                    if 'Category' in destination_info:
                        response = f"🏷️ {destination_name} is categorized as: {destination_info['Category']} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # Religion questions
                elif any(keyword in input_lower for keyword in ['religion', 'faith', 'belief', 'worship']):
                    if 'Majority Religion' in destination_info:
                        response = f"🕌 The majority religion in {destination_name} is: {destination_info['Majority Religion']} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # Safety questions
                elif any(keyword in input_lower for keyword in ['safe', 'safety', 'danger', 'dangerous', 'crime']):
                    if 'Safety' in destination_info:
                        safety_info = destination_info['Safety']
                        response = f"🛡️ Safety in {destination_name}: {safety_info} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # Language questions
                elif any(keyword in input_lower for keyword in ['language', 'speak', 'tongue', 'dialect']):
                    if 'Language' in destination_info:
                        response = f"🗣️ The official language in {destination_name} is: {destination_info['Language']} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # Region questions
                elif any(keyword in input_lower for keyword in ['region', 'area', 'continent', 'part of']):
                    if 'Region' in destination_info:
                        response = f"🌍 {destination_name} is located in: {destination_info['Region']} (Confidence: 100% - from verified dataset)"
                        return self.language_manager.translate_text(response, self.target_language)
                
                # If no specific attribute matched, return general info with high confidence
                response = self.generate_destination_response(destination_info)
                if response:
                    response += "\n(Confidence: 95% - from verified dataset)"
                    return self.language_manager.translate_text(response, self.target_language)
        
        # If no exact match, try model prediction
        selected_model = self.select_model(processed_input)
        if selected_model:
            response = self.generate_response(selected_model, processed_input)
            if response:
                return self.language_manager.translate_text(response, self.target_language)
        
        # Try GPT-3.5 fallback with medium confidence
        gpt_response = await self.get_gpt_fallback(processed_input)
        if gpt_response:
            return self.language_manager.translate_text(
                f"✈️ While I couldn't find that in my database, here's what I know (Confidence: 75% - from general knowledge):\n{gpt_response}",
                self.target_language
            )
        
        return self.language_manager.translate_text(
            "I couldn't find that information in my travel database (Confidence: 0% - no data found). Could you try asking differently?",
            self.target_language
        )
    
    async def predict_tag(self, sentence):
        """Special handler for tag prediction queries"""
        for model in self.trained_models:
            if model.get('vectorizer') and 'sentence' in model.get('features', []):
                try:
                    # Vectorize the input
                    X = model['vectorizer'].transform([sentence])
                    
                    # Make prediction
                    tag = model['model'].predict(X)[0]
                    if model.get('label_encoder'):
                        tag = model['label_encoder'].inverse_transform([tag])[0]
                    
                    # Get confidence
                    confidence = 0.5  # Default
                    if hasattr(model['model'], "predict_proba"):
                        proba = model['model'].predict_proba(X)
                        confidence = np.max(proba)
                    
                    # Get example sentences for this tag
                    examples = self.get_tag_examples(tag)
                    
                    # Build response
                    response = [f"✈️ Predicted tag for '{sentence}': **{tag}** (Confidence: {confidence:.0%})"]
                    if examples:
                        response.append("\n**Example sentences with this tag:**")
                        response.extend(f"- \"{ex}\"" for ex in examples[:3])  # Show max 3 examples
                    
                    return '\n'.join(response)
                
                except Exception as e:
                    print(f"Tag prediction error: {e}")
                    return "⚠️ Error predicting tag (Confidence: 0% - error occurred). Please try again."
        
        return "I couldn't find a tag prediction model in my database (Confidence: 0% - no model found)."

assistant = TravelModelAssistant()

@bot.event
async def on_ready():
    print(f'✅ Logged in as {bot.user} (ID: {bot.user.id})')
    print(f'✅ Loaded {len(assistant.trained_models)} travel models')
    print(f'✅ Loaded data for {len(assistant.destination_data)} destinations')
    if assistant.gpt_enabled:
        print('✅ GPT-3.5 fallback enabled')
    else:
        print('⚠️ GPT-3.5 fallback disabled (no OPENAI_API_KEY found)')

@bot.command()
async def hi(ctx):
    greetings = [
        "Hello traveler! ✈️ Ready to explore travel information?",
        "Hi there! 🌍 I specialize in travel data analysis. How can I help?",
        "Greetings! 🗺️ I can answer questions about countries, foods, and religions.",
    ]
    greeting = random.choice(greetings)
    greeting = assistant.language_manager.translate_text(greeting, assistant.target_language)
    await ctx.send(greeting)

@bot.command()
async def ask(ctx, *, question):
    try:
        response = await assistant.get_travel_response(question)
        await ctx.send(response)
    except Exception as e:
        print(f"Error in ask command: {e}")
        error_msg = "⚠️ I encountered an error processing your question. Please try again."
        await ctx.send(assistant.language_manager.translate_text(error_msg, assistant.target_language))

@bot.command()
async def models(ctx):
    if not assistant.trained_models:
        msg = "⚠️ No travel models are currently loaded."
        await ctx.send(assistant.language_manager.translate_text(msg, assistant.target_language))
        return
    
    message = ["**My Specialized Knowledge:**"]
    for idx, model_info in enumerate(assistant.trained_models, 1):
        if isinstance(model_info, dict):
            predictor = model_info.get('predictor', 'travel information')
            features = model_info.get('features', [])
            metrics = model_info.get('metrics', {})
            
            # Get the most relevant metric based on model type
            if 'accuracy' in metrics:
                performance = f"Accuracy: {metrics['accuracy']:.1%}"
            elif 'r2' in metrics:
                performance = f"R² Score: {metrics['r2']:.2f}"
            else:
                performance = "Performance data not available"
            
            model_entry = [
                f"\n**Model {idx}**: Predicts **{predictor}**",
                f"  - Features: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}",
                f"  - {performance}"
            ]
            message.extend(model_entry)
    
    await ctx.send(assistant.language_manager.translate_text('\n'.join(message), assistant.target_language))

@bot.command()
async def help(ctx):
    help_message = """
    **Travel Data Assistant Help** 🌐

    I specialize in analyzing travel-related data including:
    - Traditional foods by region
    - Country information
    - Religious demographics
    - Capital cities
    - Popular attractions

    **Commands:**
    `!hi` - Greet me
    `!ask <question>` - Ask travel data questions
    `!models` - See what travel topics I can analyze
    `!help` - Show this message
    `!setlang <code>` - Change language

    Example questions:
    `!ask country of geneva`
    `!ask what is the best place to visit?`
    `!ask nation of stonehenge`

    Example languages: 'en', 'es', 'fr', 'de', 'zh-cn', 'ja'
    """
    await ctx.send(assistant.language_manager.translate_text(help_message, assistant.target_language))

@bot.command()
async def setlang(ctx, lang_code: str):
    lang_code = lang_code.lower()
    if lang_code in SUPPORTED_LANGUAGES:
        assistant.target_language = lang_code
        assistant.language_manager.target_language = lang_code
        await ctx.send(f"Language set to {lang_code.upper()}")
    else:
        supported = ', '.join(SUPPORTED_LANGUAGES)
        await ctx.send(f"Unsupported language. Available options: {supported}")

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if TOKEN:
    bot.run(TOKEN)
else:
    print("❌ DISCORD_BOT_TOKEN is not set in .env file!")