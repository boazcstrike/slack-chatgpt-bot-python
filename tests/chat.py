import os
import json

from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from urllib.request import urlopen
import tiktoken


load_dotenv()
model = 'gpt-3.5-turbo'


def save_user_chat_history(message, parent_message_text=None, thread_ts=None):
    """this is for context windowing, to keep track of the last few messages in a channel for the chatbot to use as context."""
    file_path = f'./tmp/test-{datetime.now().strftime("%m-%d-%Y")}.txt'
    now = datetime.now().isoformat()

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_messages = [json.loads(line) for line in file]
    else:
        existing_messages = []
    print(f'Using {len(existing_messages)} messages from chat history')

    system_message = {'role': 'system', 'content': 'You are a helpful corporate assistant'}
    if system_message not in existing_messages:
        existing_messages.append(system_message)

    user_message = {'role': 'user', 'content': message}
    existing_messages.append(user_message)

    with open(file_path, 'w') as file:
        for message in existing_messages:
            file.write(json.dumps(message) + '\n')
    print('user msg written to textfile')

    return existing_messages


def save_assistant_chat_history(message, messages, thread_ts=None):
    """saves the gpt response to the chat history"""
    file_path = f'./tmp/test-{datetime.now().strftime("%m-%d-%Y")}.txt'
    now = datetime.now().isoformat()

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_messages = [json.loads(line) for line in file]
    else:
        existing_messages = []
    print(f'Using {len(existing_messages)} messages from chat history')

    existing_messages.append({'role': 'assistant', 'content': message})

    with open(file_path, 'w') as file:
        for message in existing_messages:
            file.write(json.dumps(message) + '\n')
    print('assistant msg written to textfile')

    return existing_messages


def count_tokens(messages, model=model):
    """assuming this is gpt-3.5-turbo-0613"""
    print('inside count tokens')
    print('messages:', messages)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


c = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY'),
)
while True:
    prompt = input("Enter a prompt: \n")

    messages = save_user_chat_history(prompt)
    print(count_tokens(messages))
    response = c.chat.completions.create(messages=messages, model=model, n=1)
    gpt_resp = response.choices[0].message.content.strip('\n')
    messages = save_assistant_chat_history(gpt_resp, messages)

    print(f'reply:\n',gpt_resp)

    print(f'\nModel: {response.model}\n')
    print(f'Prompt tokens: {response.usage.prompt_tokens}')
    print(f'Completion tokens: {response.usage.completion_tokens}')
    print(f'Total tokens: {response.usage.total_tokens}\n')