import discord
from discord.ext import commands
from config import TOKEN

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.load_extension('events.on_message')
bot.load_extension('events.on_member_join')

bot.run(TOKEN)
