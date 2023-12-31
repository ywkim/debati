from __future__ import annotations

import os
from abc import ABC, abstractmethod
from configparser import ConfigParser
from typing import Any

from google.cloud import firestore
from langchain.chat_models import ChatOpenAI

MAX_TOKENS = 1023


def safely_get_field(
    document: firestore.DocumentSnapshot,
    field_path: str,
    default: (Any | None) = None,
) -> Any:
    """
    Safely retrieves a value from the document snapshot of Firestore using a
    field path. Returns a default value if the field path
    does not exist within the document.

    Args:
        document (DocumentSnapshot): The snapshot of the Firestore document.
        field_path (str): A dot-delimited path to a field in the Firestore document.
        default (Optional[Any]): The default value to return if the field doesn't exist.

    Returns:
        Any: The value retrieved from the document for the field path, if it exists;
             otherwise, the default value.
    """
    try:
        value = document.get(field_path)
        if value is None:
            return default
        return value
    except KeyError:
        return default


class AppConfig(ABC):
    """
    Manages the application configuration settings.

    This class is responsible for loading configuration settings from various sources
    including environment variables, files, and Firebase Firestore.

    Attributes:
        config (ConfigParser): A ConfigParser object holding the configuration.
    """

    DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
        "settings": {
            "chat_model": "gpt-4",
            "questioning_system_prompt": "You are a helpful assistant answering student questions.",
            "pro_system_prompt": "You are a debate assistant opposing the student's viewpoint.",
            "con_system_prompt": "You are a debate assistant opposing the student's viewpoint.",
            "debate_evaluation_prompt": "Evaluate the following student's debate performance based on the coherence, relevance, and logical consistency of their arguments. Provide a score out of 10.",
            "temperature": 0,
            "vision_enabled": False,
        },
        "firebase": {"enabled": False},
        "langsmith": {"enabled": False},
        "proactive_messaging": {
            "enabled": False,
            "temperature": 1,
        },
    }

    def __init__(self):
        """Initialize AppConfig with default settings."""
        self.config: ConfigParser = ConfigParser()
        self.config.read_dict(self.DEFAULT_CONFIG)

    @property
    def debate_topic(self) -> str:
        """Returns the current debate topic."""
        return self.config.get("settings", "debate_topic")

    @property
    def questioning_system_prompt(self) -> str:
        """Returns the system prompt for questioning AI."""
        return self.config.get(
            "settings",
            "questioning_system_prompt",
            fallback=self.DEFAULT_CONFIG["settings"]["questioning_system_prompt"],
        )

    @property
    def pro_system_prompt(self) -> str:
        """Returns the system prompt for pro stance in debating AI."""
        return self.config.get(
            "settings",
            "pro_system_prompt",
            fallback=self.DEFAULT_CONFIG["settings"]["pro_system_prompt"],
        )

    @property
    def con_system_prompt(self) -> str:
        """Returns the system prompt for con stance in debating AI."""
        return self.config.get(
            "settings",
            "con_system_prompt",
            fallback=self.DEFAULT_CONFIG["settings"]["con_system_prompt"],
        )

    @property
    def debate_evaluation_prompt(self) -> str:
        """Returns the system prompt for debate evaluation."""
        return self.config.get(
            "settings",
            "debate_evaluation_prompt",
            fallback=self.DEFAULT_CONFIG["settings"]["debate_evaluation_prompt"],
        )

    @property
    def vision_enabled(self) -> bool:
        """Determines if vision (image analysis) feature is enabled."""
        return self.config.getboolean(
            "settings",
            "vision_enabled",
            fallback=self.DEFAULT_CONFIG["settings"]["vision_enabled"],
        )

    @property
    def firebase_enabled(self) -> bool:
        """Determines if Firebase integration is enabled."""
        return self.config.getboolean(
            "firebase", "enabled", fallback=self.DEFAULT_CONFIG["firebase"]["enabled"]
        )

    @property
    def langsmith_enabled(self) -> bool:
        """Determines if LangSmith feature is enabled."""
        return self.config.getboolean(
            "langsmith", "enabled", fallback=self.DEFAULT_CONFIG["langsmith"]["enabled"]
        )

    @property
    def langsmith_api_key(self) -> str:
        """Retrieves the LangSmith API key."""
        return self.config.get("langsmith", "api_key")

    @property
    def proactive_messaging_enabled(self) -> bool:
        """Determines if proactive messaging feature is enabled."""
        return self.config.getboolean(
            "proactive_messaging",
            "enabled",
            fallback=self.DEFAULT_CONFIG["proactive_messaging"]["enabled"],
        )

    @property
    def proactive_message_interval_days(self) -> float:
        """Returns the average interval in days between proactive messages."""
        return self.config.getfloat("proactive_messaging", "interval_days")

    @property
    def proactive_system_prompt(self) -> str:
        """Returns the system prompt for proactive messaging."""
        return self.config.get("proactive_messaging", "system_prompt")

    @property
    def proactive_slack_channel(self) -> str:
        """Returns the Slack channel ID where proactive messages will be posted."""
        return self.config.get("proactive_messaging", "slack_channel")

    @property
    def proactive_message_temperature(self) -> float:
        """Retrieves the temperature setting for proactive messaging."""
        return self.config.getfloat(
            "proactive_messaging",
            "temperature",
            fallback=self.DEFAULT_CONFIG["proactive_messaging"]["temperature"],
        )

    def _validate_config(self) -> None:
        """Validate that required configuration variables are present."""
        required_settings = ["openai_api_key"]
        for setting in required_settings:
            assert setting in self.config["api"], f"Missing configuration for {setting}"

        required_firebase_settings = ["enabled"]
        for setting in required_firebase_settings:
            assert (
                setting in self.config["firebase"]
            ), f"Missing configuration for {setting}"

        if self.langsmith_enabled:
            assert self.langsmith_api_key, "Missing configuration for LangSmith API key"

    def _apply_settings_from_companion(
        self, companion: firestore.DocumentSnapshot
    ) -> None:
        """
        Applies settings from the given companion Firestore document to the provided app configuration.

        Args:
            companion (firestore.DocumentSnapshot): Firestore document snapshot containing companion settings.
        """
        # Retrieve settings and use defaults if necessary
        settings: dict[str, Any] = {
            "chat_model": (
                safely_get_field(
                    companion,
                    "chat_model",
                    self.DEFAULT_CONFIG["settings"]["chat_model"],
                )
            ),
            "temperature": (
                safely_get_field(
                    companion,
                    "temperature",
                    self.DEFAULT_CONFIG["settings"]["temperature"],
                )
            ),
            "vision_enabled": (
                safely_get_field(
                    companion,
                    "vision_enabled",
                    self.DEFAULT_CONFIG["settings"]["vision_enabled"],
                )
            ),
            "debate_topic": (companion.get("debate_topic")),
            "debate_evaluation_prompt": (
                safely_get_field(
                    companion,
                    "debate_evaluation_prompt",
                    self.DEFAULT_CONFIG["settings"]["debate_evaluation_prompt"],
                )
            ),
            "questioning_system_prompt": (
                safely_get_field(
                    companion,
                    "questioning_system_prompt",
                    self.DEFAULT_CONFIG["settings"]["questioning_system_prompt"],
                )
            ),
            "pro_system_prompt": (
                safely_get_field(
                    companion,
                    "pro_system_prompt",
                    self.DEFAULT_CONFIG["settings"]["pro_system_prompt"],
                )
            ),
            "con_system_prompt": (
                safely_get_field(
                    companion,
                    "con_system_prompt",
                    self.DEFAULT_CONFIG["settings"]["con_system_prompt"],
                )
            ),
        }

        # Apply the new configuration settings
        self.config.read_dict({"settings": settings})

    def _apply_langsmith_settings(self):
        """
        Applies LangSmith settings if enabled.
        Sets LangSmith API key as an environment variable.
        """
        if self.langsmith_enabled:
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

    @abstractmethod
    def load_config(self):
        """
        Abstract method to load configuration.

        This method should be implemented in derived classes to load configurations
        from specific sources.
        """

    def get_readable_config(self) -> str:
        """
        Retrieves a human-readable string of the current non-sensitive configuration.

        Returns:
            str: A string representing the current configuration excluding sensitive details.
        """
        readable_config = (
            f"Chat Model: {self.config.get('settings', 'chat_model')}\n"
            f"Temperature: {self.config.get('settings', 'temperature')}\n"
            f"Vision Enabled: {'Yes' if self.vision_enabled else 'No'}"
        )
        return readable_config


def init_chat_model(app_config: AppConfig) -> ChatOpenAI:
    """
    Initialize the langchain chat model.

    Args:
        app_config (AppConfig): Application configuration object.

    Returns:
        ChatOpenAI: Initialized chat model.
    """
    config = app_config.config
    chat = ChatOpenAI(
        model=config.get("settings", "chat_model"),
        temperature=float(config.get("settings", "temperature")),
        openai_api_key=config.get("api", "openai_api_key"),
        openai_organization=config.get("api", "openai_organization", fallback=None),
        max_tokens=MAX_TOKENS,
    )  # type: ignore
    return chat


def init_proactive_chat_model(app_config: AppConfig) -> ChatOpenAI:
    """
    Initializes a chat model specifically for proactive messaging.

    This function creates a chat model instance using settings configured for
    proactive messaging, including the temperature setting which influences the
    creativity of the generated messages.

    Args:
        app_config (AppConfig): The configuration object containing settings
                                     for proactive messaging.

    Returns:
        ChatOpenAI: An initialized chat model for proactive messaging.
    """
    proactive_temp = app_config.proactive_message_temperature
    chat = ChatOpenAI(
        model=app_config.config.get("settings", "chat_model"),
        temperature=proactive_temp,
        openai_api_key=app_config.config.get("api", "openai_api_key"),
        openai_organization=app_config.config.get(
            "api", "openai_organization", fallback=None
        ),
        max_tokens=MAX_TOKENS,
    )  # type: ignore
    return chat
