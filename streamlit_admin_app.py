from __future__ import annotations

import csv
from io import StringIO
from typing import Any

import streamlit as st

from config.streamlit_config import StreamlitAppConfig


class StreamlitAdminApp:
    """
    A Streamlit web app for administering a chatbot application.

    This class encapsulates the functionalities required for administering
    the chatbot settings, such as updating configurations and monitoring chatbot data.

    Attributes:
        app_config (StreamlitAppConfig): The application configuration object.
        db (firestore.Client): The Firestore client for database interactions.
    """

    def __init__(self):
        """Initializes the StreamlitAdminApp with necessary configurations."""
        self.app_config = StreamlitAppConfig()
        self.app_config.load_config()
        self.db = self.app_config._initialize_firebase_client()

    def get_companion_data(self, companion_id: str) -> dict[str, str]:
        """
        Retrieves companion data from Firestore.

        Args:
            companion_id (str): The unique identifier of the companion.

        Returns:
            dict[str, str]: The data of the companion.

        Raises:
            ValueError: If the companion document does not exist.
        """
        companion_ref = self.db.collection("Companions").document(companion_id)
        companion = companion_ref.get()
        if companion.exists:
            return companion.to_dict()
        raise ValueError(f"Companion document with ID {companion_id} does not exist.")

    def upload_companion_data(
        self, companion_id: str, companion_data: dict[str, str]
    ) -> None:
        """
        Uploads companion data to Firestore.

        Args:
            companion_id (str): The unique identifier of the companion.
            companion_data (dict[str, str]): The companion data to upload.
        """
        companions_ref = self.db.collection("Companions")
        companions_ref.document(companion_id).set(companion_data)

    def get_companion_ids(self) -> list[str]:
        """
        Retrieves all companion IDs from Firestore.

        Returns:
            list[str]: A list of companion IDs.
        """
        companions_ref = self.db.collection("Companions")
        companions = companions_ref.stream()
        return [companion.id for companion in companions]


def load_prefix_messages_from_csv(csv_content: str) -> list[dict[str, str]]:
    """
    Load prefix messages from a CSV string and return them as a list of dictionaries.

    Args:
        csv_content (str): The string content of the CSV file.

    Returns:
        list[dict[str, str]]: A list of dictionaries representing the messages.

    The CSV file should contain messages with their roles ('AI', 'Human', 'System')
    and content. These roles are mapped to Firestore roles ('assistant', 'user', 'system').
    """
    role_mappings = {"AI": "assistant", "Human": "user", "System": "system"}

    messages: list[dict[str, str]] = []

    reader = csv.reader(StringIO(csv_content))

    for row in reader:
        role, content = row
        if role not in role_mappings:
            raise ValueError(
                f"Invalid role '{role}' in CSV content. Must be one of {list(role_mappings.keys())}."
            )

        firestore_role = role_mappings[role]
        messages.append({"role": firestore_role, "content": content})

    return messages


def format_prefix_messages_for_display(messages: list[dict[str, str]]) -> str:
    """
    Formats the prefix messages as a string for display in a text area.

    Args:
        messages (list[dict[str, str]]): The list of prefix messages.

    Returns:
        str: The formatted string representation of the messages in CSV format.
    """
    output = StringIO()
    writer = csv.writer(output)
    role_mappings = {"assistant": "AI", "user": "Human", "system": "System"}

    for message in messages:
        # Convert Firestore role back to CSV role
        csv_role = role_mappings.get(message["role"], message["role"])
        writer.writerow([csv_role, message["content"]])

    return output.getvalue().strip()


def main():
    """
    The main function to run the Streamlit admin app.

    This function sets up the Streamlit interface and handles user interactions
    for administering chatbot configurations.
    """
    st.title("Admin")

    admin_app = StreamlitAdminApp()

    chat_models = ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo"]

    # Companion ID selection and existing data pre-fill logic
    companion_ids = admin_app.get_companion_ids()
    new_companion_option = "Add New Companion"
    selected_companion_id = st.selectbox(
        "Select Companion ID", [new_companion_option] + companion_ids
    )

    # Handling new companion ID input
    companion_id_to_upload = None
    existing_data: dict[str, Any] = {}
    if selected_companion_id == new_companion_option or selected_companion_id is None:
        companion_id_to_upload = st.text_input("Enter New Companion ID")
        existing_data = {}
    else:
        companion_id_to_upload = selected_companion_id
        existing_data = admin_app.get_companion_data(selected_companion_id)

    # Adjust the chat_models list based on existing data
    existing_model = existing_data.get("chat_model", "gpt-4")
    if existing_model not in chat_models:
        chat_models.append(existing_model)
    chat_model_index = chat_models.index(existing_model)

    chat_model = st.selectbox("Chat Model", chat_models, index=chat_model_index)

    # 새로운 토론 관련 설정 추가
    debate_topic = st.text_input(
        "Debate Topic", value=existing_data.get("debate_topic", "")
    )
    debate_evaluation_prompt = st.text_area(
        "Debate Evaluation Prompt",
        value=existing_data.get("debate_evaluation_prompt", ""),
    )
    questioning_system_prompt = st.text_area(
        "Questioning System Prompt",
        value=existing_data.get("questioning_system_prompt", ""),
    )
    pro_system_prompt = st.text_area(
        "Pro System Prompt", value=existing_data.get("pro_system_prompt", "")
    )
    con_system_prompt = st.text_area(
        "Con System Prompt", value=existing_data.get("con_system_prompt", "")
    )

    # Temperature 및 Prefix Messages 설정
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        value=existing_data.get("temperature", 1.0),
    )

    # Companion data upload logic
    if companion_id_to_upload and st.button("Upload Companion Data"):
        companion_data = {
            "chat_model": chat_model,
            "temperature": temperature,
            "vision_enabled": False,
            "debate_topic": debate_topic,
            "debate_evaluation_prompt": debate_evaluation_prompt,
            "questioning_system_prompt": questioning_system_prompt,
            "pro_system_prompt": pro_system_prompt,
            "con_system_prompt": con_system_prompt,
        }
        admin_app.upload_companion_data(companion_id_to_upload, companion_data)
        st.success(f"Companion '{companion_id_to_upload}' data updated successfully.")


if __name__ == "__main__":
    main()
