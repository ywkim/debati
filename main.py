import discord
from discord.ext import commands
from config import TOKEN

intents = discord.Intents.default()  # 기본적인 모든 intents를 활성화합니다.
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.load_extension('commands')
bot.run(TOKEN)
