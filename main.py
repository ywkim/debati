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
