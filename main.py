import configparser
import os

import openai
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.utilities import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from serp_loader import SerpAPILoader
from search_qa import SearchQA
from web_qa import WebQA
from git_qa import GitQA
from book_qa import BookQA
import argparse
import json
import pinecone

import interactions
from interactions import OptionType, Option, CommandContext
import logging

DEFAULT_CONFIG = {
    "settings": {
        "chat_model": "gpt-4",
        "system_prompt": "You are a helpful assistant.",
        "temperature": "0",
    },
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config():
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read("config.ini")
    return config

def load_tools(config):
    llm = OpenAI(temperature=0, openai_api_key=config.get("api", "openai_api_key"))
    serp = SerpAPIWrapper(serpapi_api_key=config.get("api", "serpapi_api_key"))
    embeddings = OpenAIEmbeddings(openai_api_key=config.get("api", "openai_api_key"), disallowed_special=())
    pinecone_index = config.get("settings", "pinecone_index")
    pinecone.init(api_key=config.get("api", "pinecone_api_key"),
                  environment=config.get("api", "pinecone_env"))
    return [
        SearchQA(llm=llm, serp=serp, embeddings=embeddings),
        WebQA(llm=llm, embeddings=embeddings, handle_tool_error=True),
        GitQA(llm=llm, embeddings=embeddings, pinecone_index=pinecone_index, handle_tool_error=True),
        BookQA(llm=llm, embeddings=embeddings, pinecone_index=pinecone_index, handle_tool_error=True),
    ]

def init_agent_with_tools():
    config = load_config()
    system_prompt = SystemMessage(content=config.get("settings", "system_prompt"))
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_prompt,
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    chat = ChatOpenAI(
        model=config.get("settings", "chat_model"),
    temperature=float(config.get("settings", "temperature")),
        openai_api_key=config.get("api", "openai_api_key"),
    )
    tools = load_tools(config)
    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    return agent

client = interactions.Client(token=os.getenv('TOKEN'), default_scope=os.getenv('GUILD_ID'))

@client.event
async def on_ready():
    logging.info(f'{client.me} is online and ready to answer your questions!')

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
    agent = init_agent_with_tools()
    response_message = agent.run(prompt)
    await ctx.send(response_message, ephemeral=(ephemeral == 'Enable'))

def main():
    parser = argparse.ArgumentParser(description='Run agent with given messages.')
    parser.add_argument('filename', type=str, help='Path to the message file')

    args = parser.parse_args()

    client.start()

if __name__ == "__main__":
    main()
