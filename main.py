import openai
import os
import random
import interactions
from interactions import OptionType, Option, CommandContext
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

client = interactions.Client(token=os.getenv('TOKEN'), default_scope=os.getenv('GUILD_ID'))

@client.event
async def on_ready():
    logging.info(f'{client.me} is online and ready to answer your questions!')

openai.api_key = os.getenv('OPENAI_API_KEY')

@client.command(
    name="ask",
    description="Answers your questions!",
    options=[
        Option(
            name="prompt",
            description="What is your question?",
            type=OptionType.STRING,
            required=True
        ),
        Option(
            name="model",
            description="What model do you want to ask from? (Default: ChatGPT)",
            type=OptionType.STRING,
            required=False,
            choices=[
                {"name": 'ChatGPT (BEST OF THE BEST)', "value": 'chatgpt'},
                {"name": 'Davinci (Most powerful)', "value": 'davinci'},
                {"name": 'Curie', "value": 'curie'},
                {"name": 'Babbage', "value": 'babbage'},
                {"name": 'Ada (Fastest)', "value": 'ada'}
            ]
        ),
        Option(
            name='ephemeral',
            description='Hides the bot\'s reply from others. (Default: Disable)',
            type=OptionType.STRING,
            required=False,
            choices=[
                {"name": 'Enable', "value": 'Enable'},
                {"name": 'Disable', "value": 'Disable'}
            ]
        )
    ]
)
async def _ask(ctx: CommandContext, prompt: str, model: str = 'chatgpt', ephemeral: str = 'Disable'):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    await ctx.send(response['choices'][0]['message']['content'], ephemeral=(ephemeral == 'Enable'))

client.start()
