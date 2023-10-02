import argparse
import asyncio
import configparser
import json
import logging

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

import pinecone
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.utilities import SerpAPIWrapper

from qa.book_qa import BookQA
from qa.git_qa import GitQA
from qa.search_qa import SearchQA
from qa.web_qa import WebQA

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


def load_config(config_file: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(config_file)
    return config


def load_tools(config: configparser.ConfigParser):
    llm = ChatOpenAI(
        model=config.get("settings", "chat_model"),
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


def init_agent_with_tools(config: configparser.ConfigParser) -> AgentExecutor:
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


def register_events_and_commands(socket_mode_client, config):
    @socket_mode_client.socket_mode_request_listeners.append
    async def handle_message(client: SocketModeClient, request: SocketModeRequest):
        if request.type != 'message':
            logging.info('Ignoring non-message event')
            return

        message_payload = request.payload["event"]
        channel_id = message_payload["channel"]

        if "bot_id" in message_payload:
            logging.info('Ignoring bot message')
            return

        text = message_payload.get("text", "")
        logging.info('Received a message: %s', text)
        response_message = await ask_question_to_agent(text, config)
        logging.info('Generated response: %s', response_message)
        await client.web_client.chat_postMessage(channel=channel_id, text=response_message)
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=request.envelope_id))

async def ask_question_to_agent(message: str, config):
    """Pass the message to the agent and get an answer."""
    agent = init_agent_with_tools(config)  # Create agent
    return await agent.arun(message)  # Get agent to process the message

async def process_messages_from_file(file_path, config):
    agent = init_agent_with_tools(config)
    with open(file_path, "r", encoding="utf-8") as message_file:
        messages = json.load(message_file)
        for message in messages:
            response_message = await agent.arun(message)
            print(response_message)


async def main():
    logging.info('Starting bot')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="config.ini",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--message_file",
        help="Path to a JSON file containing messages to process.",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    if args.message_file is not None:
        await process_messages_from_file(args.message_file, config)
    else:
        slack_bot_token = config.get("api", "slack_bot_token")
        slack_app_token = config.get("api", "slack_app_token")

        logging.info('Initializing Slack client')
        web_client = AsyncWebClient(token=slack_bot_token)

        logging.info('Initializing Socket Mode client')
        socket_mode_client = SocketModeClient(app_token=slack_app_token, web_client=web_client)

        logging.info('Registering event and command handlers')
        register_events_and_commands(socket_mode_client, config)

        logging.info('Starting Socket Mode client')
        await socket_mode_client.connect()

        logging.info('Bot is running')

        # Just not to stop this process
        await asyncio.sleep(float("inf"))

if __name__ == "__main__":
    asyncio.run(main())
