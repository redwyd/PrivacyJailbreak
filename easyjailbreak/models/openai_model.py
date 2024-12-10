import logging
import warnings
from .model_base import BlackBoxModelBase
from openai import OpenAI
from fastchat.conversation import get_conv_template
from httpx import URL


class OpenaiModel(BlackBoxModelBase):
    def __init__(self, model_name: str, api_keys: str, base_url: str, generation_config=None):
        """
        Initializes the OpenAI model with necessary parameters.
        :param str model_name: The name of the model to use.
        :param str api_keys: API keys for accessing the OpenAI service.
        :param str template_name: The name of the conversation template, defaults to 'chatgpt'.
        :param dict generation_config: Configuration settings for generation, defaults to an empty dictionary.
        :param str|URL base_url: The base URL for the OpenAI API, defaults to None.
        """
        self.client = OpenAI(api_key=api_keys, base_url=base_url)
        self.model_name = model_name
        self.conversation = get_conv_template('chatgpt')
        self.generation_config = generation_config if generation_config is not None else {}

        self.base_url = base_url

    def set_system_message(self, system_message: str):
        """
        Sets a system message for the conversation.
        :param str system_message: The system message to set.
        """
        self.conversation.system_message = system_message

    def generate(self, messages, clear_old_history=True, **kwargs):
        """
        Generates a response based on messages that include conversation history.
        :param list[str]|str messages: A list of messages or a single message string.
                                       User and assistant messages should alternate.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The response generated by the OpenAI model based on the conversation history.
        """
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        messages = self.conversation.to_openai_api_messages()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
            **self.generation_config
        )
        return response.choices[0].message.content

    def batch_generate(self, conversations, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :return list[str]: A list of responses for each conversation.
        """
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn(
                    'For batch generation based on several conversations, provide a list[str] for each conversation. '
                    'Using list[list[str]] will avoid this warning.')
            responses.append(self.generate(conversation, **kwargs))
        return responses
