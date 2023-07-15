import os
import random
import discord
from discord.ext import commands
from discord_slash import SlashCommand, SlashContext
from discord_slash.utils.manage_commands import create_option

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)
slash = SlashCommand(bot, sync_commands=True)

EMOJIS = ["😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇"]

@slash.slash(
    name="test",
    description="A test command",
    options=[
        create_option(
            name="emoji",
            description="Choose an emoji",
            option_type=3,
            required=False,
            choices=[{"name": emoji, "value": emoji} for emoji in EMOJIS]
        )
    ]
)
async def _test(ctx: SlashContext, emoji: str = None):
    if not emoji:
        emoji = random.choice(EMOJIS)
    await ctx.send(content=f"Hello world {emoji}", hidden=False)

RPS_CHOICES = ["rock", "paper", "scissors"]
@slash.slash(
    name="challenge",
    description="Challenge to a match of rock paper scissors",
    options=[
        create_option(
            name="object",
            description="Pick your object",
            option_type=3,
            required=True,
            choices=[{"name": choice.capitalize(), "value": choice} for choice in RPS_CHOICES]
        )
    ]
)
async def _challenge(ctx: SlashContext, object: str):
    bot_choice = random.choice(RPS_CHOICES)
    if object == bot_choice:
        result = "It's a draw!"
    elif (object == "rock" and bot_choice == "scissors") or (object == "paper" and bot_choice == "rock") or (object == "scissors" and bot_choice == "paper"):
        result = "You win!"
    else:
        result = "You lose!"
    await ctx.send(content=f"You chose {object}, I chose {bot_choice}. {result}", hidden=False)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.load_extension('commands.ask')
bot.load_extension('commands.imagine')
bot.load_extension('commands.optimize')
bot.load_extension('commands.translate')
bot.load_extension('events.on_message')
bot.load_extension('events.on_member_join')

bot.run(os.getenv('TOKEN'))
