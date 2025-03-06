import random
import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
# from utils import predict_response  # Import NLP function

# Load .env file
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True  # Enable member detection


bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# When the bot start to run
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID : {bot.user.id})")

# @bot.event
# async def on_message(message):
#     if message.author == bot.user:
#         return
    
#     # Let commands work normally
#     if message.content.startswith("!"):
#         await bot.process_commands(message)
#         return

#     response = predict_response(message.content)
#     await message.channel.send(response)


# Command to manually trigger NLP responses
# @bot.command()
# async def ask(ctx, *, question):
#     response = predict_response(question)
#     await ctx.send(response)

@bot.command()
async def hi(ctx, *args):  
    response_list = ["hi", "hello"]

    sender = ctx.author  # Get user who sent the message
    sender_mention = sender.mention  # Get @mention format

    # Join all arguments into a single string (if any words exist)
    user_message = " ".join(args) if args else "(no message provided)"

    # Print in terminal
    print(f"[Terminal] Sender: {sender.display_name}, Message: {user_message}")

    # Bot's response in Discord
    response = f"{random.choice(response_list)} {sender_mention}, Why are u gay?\n**Message:** `{user_message}`"
    await ctx.send(response)
    
    
    
    
    

# Read token from .env
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Run bot
if TOKEN:
    bot.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN is not set in .env file!")
