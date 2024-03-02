import json
import os
import random
import re

from dotenv import load_dotenv
from datetime import datetime, timedelta
from urllib.request import urlopen

from slack_bolt import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.socket_mode import SocketModeHandler

from utils.core import get_env, log, validate_input
from utils.home import build_app_home_blocks
from utils.api.openai import OpenAIAPI
from utils.messages.main import bot_busy_messages


class SlackBot():
    """the slack bot class."""
    def __init__(self):
        print('starting slack bot connection...')
        load_dotenv()
        self.token = os.getenv('SLACK_BOT_USER_OAUTH_TOKEN')
        self.signing_secret = os.getenv('SLACK_BOT_SIGNING_SECRET')
        self.app_token = os.getenv('SLACK_BOT_APP_LEVEL_TOKEN')
        self.client = WebClient(self.token)
        self.bot_id = self.client.auth_test().data['user_id']
        self.bot_name = self.client.auth_test().data['user']

        self.bot_channel = 'ai-dump'
        self.dm_channel_ids = {}
        self.last_request_datetime = {}
        self.busy_messages = bot_busy_messages
        self.settings = {
        "history_expires_seconds": get_env('HISTORY_EXPIRES_IN', '900'),
        "history_size": get_env('HISTORY_SIZE', '10'),
        }
        self.image_base_url = './tmp/'

        self.app = App(
            token=self.token,
            signing_secret=self.signing_secret,
        )
        self.register_listeners()


    def start_bot(self) -> None:
        """starts the slack bot and listens for incoming messages."""
        self.set_status(
            status_text='Ask me anything in #api-dump', status_emoji=':loading_apple:')
        print(f'\n\033[1mStarting Slackbot Version 0.2.1b\033[0m\nby \033[1mBoaz Sze @ {datetime.now().year}\033[0m\n')
        try:
            SocketModeHandler(self.app, self.app_token).start()
        except KeyboardInterrupt:
            log('Stopping server... please wait...')
        self.set_status(status_text='I am offline.', status_emoji=':zzz:')

    def set_status(self, status_text='', status_emoji=':blush:') -> None:
        """
        https://api.slack.com/methods/users.profile.set
        updates the bot's status
        """
        try:
            self.client.users_profile_set(
                profile={"status_text": status_text , "status_emoji": status_emoji, "status_expiration": 0})
            log("Status updated successfully!")
        except SlackApiError as e:
            log(f"Error updating status: {e.response['error']}")

    def send_message(self, channel, message, blocks=None, thread_ts=None, reply_broadcast=None) -> None:
        """sends a message to a slack channel."""
        post_message = {
        "channel": channel,
        "text": message,
        }

        if blocks:
            post_message.update({"blocks": blocks})
        if thread_ts:
            post_message.update({"thread_ts": thread_ts})
        if reply_broadcast:
            post_message.update({"reply_broadcast": reply_broadcast})
        try:
            self.client.chat_postMessage(**post_message)
        except SlackApiError as e:
            log(f"Error sending message: {e.response['error']}")

    def update_app_home(self, body, logger):
        user_id = body["event"]["user"]
        blocks = build_app_home_blocks()
        try:
            self.client.views_publish(
            user_id=user_id,
            view={
            "type": "home",
            "blocks": blocks
            }
        )
        except SlackApiError as e:
            logger.error(f"Error publishing App Home: {e}")

    def update_home_tab(self, event, logger):
        try:
            self.client.views_publish(
            user_id=event["user"],
            view={
            "type": "home",
            "blocks": build_app_home_blocks(),
            },
        )
        except Exception as e:
            logger.error(f"Error publishing home tab: {e}")

    def register_listeners(self):
        @self.app.event("app_mention")
        def handle_app_mentions(body, say, payload):
            print(body)
            try:
                message = str(body['event']['text']).strip()
                channel = body['event']['channel']
                user = body['event']['user']
                thread_ts = body['event']['thread_ts'] if 'thread_ts' in body['event'] else None

                self.handle_msg(message, channel, user, thread_ts, direct_message=True)
            except Exception as e:
                say(f"Error handling app mention: {e}")
                log(f'Error handling app mention: {e}', error=True)

        @self.app.event("reaction_added")
        def handle_reaction_added_events(body, logger):
            print('someone reacted')

        @self.app.event("message")
        def handle_message_events(body, logger):
            pass

        @self.app.event("file_shared")
        def handle_file_shared_events(body, logger):
            pass

    def handle_msg(self, message, user, channel, thread_ts=None, direct_message=False, in_thread=False):
        if channel not in self.last_request_datetime:
            self.last_request_datetime[channel] = datetime.fromtimestamp(0)
        # Let the user know that we are busy with the request if enough time has passed since last message
        if self.last_request_datetime[channel] + timedelta(seconds=int(self.settings['history_expires_seconds'])) < datetime.now():
            self.send_message(
                channel=channel,
                thread_ts=thread_ts,
                message=random.choice(self.busy_messages))

        self.last_request_datetime[channel] = datetime.now()

        # Read parent message content if called inside thread conversation
        parent_message_text = None
        if thread_ts and not direct_message:
            conversation = self.client.conversations_replies(
                channel=channel, ts=thread_ts)
            if len(conversation['messages']) > 0 and validate_input(conversation['messages'][0]['text']):
                parent_message_text = conversation['messages'][0]['text']

        message = re.sub(r'<[^>]*>', '', message, count=1)
        prompt = message.split('@', 1)[0].strip()

        if len(prompt.strip()) == 0 and parent_message_text is None:
            return
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if prompt.lower().startswith('image: ') or prompt.lower().startswith('img: '):
            self.handle_image_generation_prompt(
                channel,
                user,
                prompt,
                parent_message_text,
                thread_ts,
                'dall-e-2',
            )

        elif prompt.lower().startswith('imagehd: ') or prompt.lower().startswith('imghd: '):
            self.handle_image_generation_prompt(
                channel,
                user,
                prompt,
                parent_message_text,
                thread_ts,
                'dall-e-3',
            )

        elif prompt.lower().startswith('imagevar: ') or prompt.lower().startswith('imgvar: '):
            self.handle_image_variation_generation_prompt(
                channel,
                user,
                prompt,
                thread_ts
            )

        elif prompt.lower().startswith('tts: '):
            self.handle_tts_prompt(
                channel,
                user,
                prompt,
                thread_ts
            )

        else:
            self.handle_chat_prompt(
                channel,
                user,
                prompt,
                parent_message_text,
                thread_ts,
                direct_message,
                in_thread
            )

    def _clean_image_prompt_message(self, message, model):
        """
        Cleans the image prompt message by removing the image: prefix and any leading/trailing whitespace.
        """
        if model == 'dall-e-2':
            return message[6:].strip(',/_')
        elif model == 'dall-e-3':
            return message[8:].strip(',/_')
        return message[6:].strip(',/_')

    def handle_image_generation_prompt(
        self, channel, user,
        message, parent_message_text, thread_ts=None, model='dall-e-2'):
        """image generation prompts goes here"""
        prompt = self._clean_image_prompt_message(message, model)
        self.send_message(
        channel=channel,
        message=random.choice(self.busy_messages) + f" generating image for your request '{prompt[:24]}'..., please wait for me! :blobcatroll:"
        )

        if parent_message_text:
            prompt = f'{parent_message_text}. {prompt}'
            log('Using parent message inside thread')

        # validate prompt
        if len(prompt) == 0:
            return self.send_message(
                channel=channel,
                message='Please check your input. To generate image use this format -> @tagme! image: robot walking a dog')

        oa = OpenAIAPI()

        image_url = oa.generate_image(
        prompt=prompt,
        model=model,
        slack=self)
        image_path = None

        try:
            image_content = urlopen(image_url).read()

            short_prompt = prompt[50:]
            image_name = f"{short_prompt.replace(' ', '_')}.png"
            image_path = f'{self.image_base_url}{image_name}'

            image_file = open(image_path, 'wb')
            image_file.write(image_content)
            image_file.close()

            upload_response = self.client.files_upload_v2(
                channel=self.dm_channel_ids.get(user, user),
                thread_ts=thread_ts,
                title=prompt,
                filename=image_name,
                file=image_path,
            )

            self.send_message(
                channel=channel,
                message=f"File url in case you need it! :blob_excited: {upload_response['file']['url_private']}")
        except SlackApiError as e:
            msg = f'Slack API error: {e}'
            log(msg, error=True)
            self.send_message(
                channel=channel,
                message=msg)

        # uncomment to save storage space
        # Remove temp image
        # if image_path and os.path.exists(image_path):
        #    os.remove(image_path)

    def handle_image_variation_generation_prompt(self, channel, user, message, attachments, thread_ts=None):
        """Handle user's slack message with the attached image"""
        pass
        # if not "attachments" in message:
        #   msg = 'No attachments found in message'
        #   self.send_message(channel=channel, thread_ts=thread_ts, message=msg + ' :sob: Please send an image together with the prompt.', reply_broadcast=thread_ts)
        #   return log(msg)

        # attachments = message["attachments"]
        # for attachment in attachments:
        #   if attachment["type"] == "image":
        #     log('Received an image, attempting to generate 2 variations')
        #     image_url = attachment["image_url"]
        #     response = self.client.files_download(url=image_url)
        #     if not response.status_code == 200:
        #       self.send_message(channel=channel, thread_ts=thread_ts, message='Something went wrong downloading the image. :sob:', reply_broadcast=thread_ts)
        #       return log("Failed to download image")

        #   else:
        #     msg = 'Attached file is not an image.'
        #     self.send_message(channel=channel, thread_ts=thread_ts, message=msg + ' :sob:', reply_broadcast=thread_ts)
        #     return log(msg)

        # image_name = os.path.basename(image_url)
        # image_path = os.path.join("./tmp", image_name)

        # with open(image_path, "wb") as f:
        #     f.write(image_response.content)

        # # todo: compress the image to handle big files
        # # processed_image = process_image(image_path)
        # # response_message = generate_response(processed_image)

        # image_name = f"{image_url.replace(' ', '_')}.png"
        # image_name = image_name.split(' /,')

        # image_path = f'{self.image_base_url}{image_name}'

        # image_file = open(image_path, 'wb')
        # image_file.write(image_content)
        # image_file.close()

        # response = client.images.create_variation(
        #   image=open("image_edit_original.png", "rb"),
        #   n=2,
        #   size="1024x1024"
        # )

        # self.send_message(channel=channel, thread_ts=thread_ts, message=response_message)
        # log(f'Response sent to user {user}: {response_message}')

    def _save_chat_history(self, channel, message, parent_message_text, thread_ts):
        """this is for context windowing, to keep track of the last few messages in a channel for the chatbot to use as context."""
        file_path = f'./tmp/{channel}.txt'

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_messages = [json.loads(line) for line in file]
        else:
            existing_messages = []

        log(f'Using {len(existing_messages)} messages from chat history')

        if parent_message_text:
            existing_messages.append({'role': 'user', 'content': parent_message_text})
            log(f'Adding parent message from thread with timestamp: {thread_ts}')

        messages = [
            {'role': 'user', 'content': message},
            {'role': 'system', 'content': 'You are a helpful assistant'}
        ]
        messages.extend(existing_messages)

        with open(file_path, 'w') as file:
            for msg in messages:
                file.write(json.dumps(msg) + '\n')

        return messages

    def handle_size(self, messages, thread_ts=None): #! unused
        """Remove the oldest 2 history message if the channel history size is exceeded for the current threa"""
        if len(list(filter(lambda x: x['thread_ts'] == thread_ts, messages))) >= (int(self.settings['history_size']) + 1) * 2:
        # Create iterator for chat history list
            chat_history_list = (msg for msg in messages if msg['thread_ts'] == thread_ts)
            first_occurance = next(chat_history_list, None)
            second_occurance = next(chat_history_list, None)
        else:
            first_occurance = None
            second_occurance = None

        # Remove first occurance
        if first_occurance:
            messages.remove(first_occurance)

        # Remove second occurance
        if second_occurance:
            messages.remove(second_occurance)

    def handle_chat_prompt(
        self, channel, user, message, parent_message_text, thread_ts=None, direct_message=False, in_thread=False):
        """basic chat completion goes here"""
        messages = self._save_chat_history(channel, message, parent_message_text, thread_ts)

        oa = OpenAIAPI()
        now = datetime.now()

        try:
            response = oa.create_completion(messages=messages)
        except Exception as e:
            log(f'ChatGPT response error: {e}', error=True)
            self.send_message(channel=channel, thread_ts=thread_ts, message=str(e))
            return

        gpt_resp = response.choices[0].message.content.strip('\n')

        messages.append({'role': 'user', 'content': message, 'created_at': now, 'thread_ts': thread_ts})
        messages.append(
        {'role': 'assistant', 'content': gpt_resp, 'created_at': now, 'thread_ts': thread_ts})

        # todo: handle size so we can save some storage space and reset the contexts
        # self.handle_size(x, messages, thread_ts)

        if direct_message:
            target_channel = self.dm_channel_ids.get(user, user)
        else:
            target_channel = channel

        self.send_message(channel=target_channel, thread_ts=thread_ts, message=gpt_resp, reply_broadcast=in_thread)

        log(f'ChatGPT response: {gpt_resp}')

    def _clean_tts_prompt_message(self, prompt_message):
        """
        Cleans the image prompt message by removing the image: prefix and any leading/trailing whitespace.
        """
        return prompt_message[3:].strip(',/_')

    def handle_tts_prompt(self, channel, user, message, thread_ts=None):
        """text-to-speech goes here"""
        current_datetime = datetime.now()
        prompt = self._clean_tts_prompt_message(message)
        log(f'Attempting to generate text-to-speech for "{prompt[:16]}..." at {current_datetime}')

        oa = OpenAIAPI()
        file_path, file_name = oa.generate_tts(prompt, user, thread_ts)

        try:
            _ = self.client.files_upload_v2(
                channel=self.dm_channel_ids.get(user, user),
                thread_ts=thread_ts,
                file=file_path,
                filename=file_name,
                title=prompt[:55],
            )
            print("Audio sent successfully to Slack!")
        except SlackApiError as e:
            print(f"Error sending audio to Slack: {e.response['error']}")