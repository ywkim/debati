import discord
from discord_slash import SlashCommand, SlashContext
from discord.ext import commands
from config import TOKEN, CLIENT_ID

bot = commands.Bot(command_prefix="!")
slash = SlashCommand(bot, sync_commands=True)

bot.load_extension('commands.ask')
bot.load_extension('commands.imagine')
bot.load_extension('commands.optimize')
bot.load_extension('commands.translate')

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.run(TOKEN)
