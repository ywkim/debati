import argparse
import asyncio
import configparser
import json
import logging
import os

import interactions
import pinecone
from interactions import CommandContext, Option, OptionType
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.utilities import SerpAPIWrapper

from book_qa import BookQA
from git_qa import GitQA
from search_qa import SearchQA
from web_qa import WebQA

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


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(config_file)
    return config


def load_tools(config):
    llm = ChatOpenAI(
        model=config.get("settings", "tool_chat_model"),
        temperature=0,
        openai_api_key=config.get("api", "openai_api_key"),
    )
    serp = SerpAPIWrapper(serpapi_api_key=config.get("api", "serpapi_api_key"))
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.get("api", "openai_api_key"), disallowed_special=()
    )
    pinecone_index = config.get("settings", "pinecone_index")
    pinecone.init(
        api_key=config.get("api", "pinecone_api_key"),
        environment=config.get("api", "pinecone_env"),
    )
    return [
        SearchQA(llm=llm, serp=serp, embeddings=embeddings),
        WebQA(llm=llm, embeddings=embeddings, handle_tool_error=True),
        GitQA(
            llm=llm,
            embeddings=embeddings,
            pinecone_index=pinecone_index,
            handle_tool_error=True,
        ),
        BookQA(
            llm=llm,
            embeddings=embeddings,
            pinecone_index=pinecone_index,
            handle_tool_error=True,
        ),
    ]


def init_agent_with_tools(config):
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


def register_events_and_commands(client, config):
    @client.event
    async def on_ready():
        logging.info("%s is online and ready to answer your questions!", client.me)

    @client.command(
        name="ask",
        description="Answers your questions!",
        options=[
            Option(
                name="prompt",
                description="What is your question?",
                type=OptionType.STRING,
                required=True,
            ),
            Option(
                name="model",
                description="What model do you want to ask from? (Default: ChatGPT)",
                type=OptionType.STRING,
                required=False,
                choices=[
                    {"name": "ChatGPT (BEST OF THE BEST)", "value": "chatgpt"},
                    {"name": "Davinci (Most powerful)", "value": "davinci"},
                    {"name": "Curie", "value": "curie"},
                    {"name": "Babbage", "value": "babbage"},
                    {"name": "Ada (Fastest)", "value": "ada"},
                ],
            ),
            Option(
                name="ephemeral",
                description="Hides the bot's reply from others. (Default: Disable)",
                type=OptionType.STRING,
                required=False,
                choices=[
                    {"name": "Enable", "value": "Enable"},
                    {"name": "Disable", "value": "Disable"},
                ],
            ),
        ],
    )
    async def _ask(
        ctx: CommandContext,
        prompt: str,
        model: str = "chatgpt",
        ephemeral: str = "Disable",
    ):
        await ctx.defer()
        agent = init_agent_with_tools(config)
        response_message = await agent.arun(prompt)
        await ctx.send(response_message, ephemeral=(ephemeral == "Enable"))


async def process_messages_from_file(file_path, config):
    agent = init_agent_with_tools(config)
    with open(file_path, "r", encoding="utf-8") as message_file:
        messages = json.load(message_file)
        for message in messages:
            response_message = await agent.arun(message)
            print(response_message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="config.ini", help="The path to a config file."
    )
    parser.add_argument(
        "--message_file", help="The path to a JSON file containing messages to process."
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    if args.message_file:
        asyncio.run(process_messages_from_file(args.message_file, config))
    else:
        default_scope = None
        if config.get("settings", "guild_id") is not None:
            default_scope = int(config.get("settings", "guild_id"))
        client = interactions.Client(
            token=config.get("api", "discord_token"), default_scope=default_scope
        )
        register_events_and_commands(client, config)
        client.start()


if __name__ == "__main__":
    main()
