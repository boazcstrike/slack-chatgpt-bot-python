import os
import random
import sys
import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import urlopen
from dotenv import load_dotenv

from openai import OpenAI

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from home import build_app_home_blocks
from utils.core import robot_working_messages
import os

dm_channel_ids = {}

def generate_image(
		prompt,
		size="1024x1024",
		quality="standard",
		channel=None,
		thread_ts=None):
	# Call the DALL-E API to generate the image (replace with actual API call)
	log(f'Attempting to generate "{prompt}" at {quality} {size}')

	# dall-e-2, dall-e-3
	model = 'dall-e-3'

	if model == 'dall-e-3':
		size = "1024x1024"
	if model == 'dall-e-2':
		size = "256x256"

	try:
		response = openai_client.images.generate(
			model=model,
			prompt=prompt,
			size=size,
			quality=quality,
			n=1
		)
		image_url = response.data[0].url
	except Exception as e:
		log(f'ChatGPT response error: {e}', error=True)
		if not channel:
			return
		slack_client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=str(e))
		return
	return image_url


def validate_input_none_or_empty(value: Optional[str]) -> bool:
	return value is not None and value.strip() != ''


def get_env(key: str, default: Optional[str]) -> str:
	value = os.getenv(key, default)
	if not validate_input_none_or_empty(value):
		value = default
	return value


def log(content: str, error: bool = False):
	now = datetime.now()
	print(f'[{now.isoformat()}] {content}', flush=True, file=sys.stderr if error else sys.stdout)

print('loading envs...')
load_dotenv()
SLACK_BOT_TOKEN = get_env('SLACK_BOT_USER_OAUTH_TOKEN', None)
SLACK_BOT_SIGNING_SECRET = get_env('SLACK_BOT_SIGNING_SECRET', None)
SLACK_APP_TOKEN = get_env('SLACK_BOT_APP_LEVEL_TOKEN', None)
OPENAI_API_KEY = get_env('OPENAI_API_KEY', None)
gptmodel = get_env('GPT_MODEL', 'gpt-4')
system_desc = get_env('GPT_SYSTEM_DESC', 'Helpful AI assistant.')
# Image size can be changed here, Must be one of 256x256, 512x512, or 1024x1024.
image_size = get_env('GPT_IMAGE_SIZE', '512x512')

print('connecting slack...')
app = App(
	token=SLACK_BOT_TOKEN,
	signing_secret=SLACK_BOT_SIGNING_SECRET,
)
slack_client = WebClient(SLACK_BOT_TOKEN)
bot_user_id = slack_client.auth_test()["user_id"]
print('done connecting slack...')

print('connecting openai_client...')
openai_client = OpenAI(
	api_key=OPENAI_API_KEY,
)
print('done connecting openai_client!')


# Function to generate a response using GPT-4
def generate_response(prompt):
    response = openai_client.chat.completions.create(
        model=gptmodel,
        messages=[{"role": "system", "content": system_desc}, {"role": "user", "content": prompt}],
        max_tokens=150,
        n=1,
        temperature=0.9,
    )
    return response.choices[0].message['content'].strip()

# Keep chat history to provide context for future prompts
chat_history = {
	'general': []
}
history_expires_seconds = int(get_env('HISTORY_EXPIRES_IN', '900'))  # 15 minutes
history_size = int(get_env('HISTORY_SIZE', '3'))

# Keep timestamps of last requests per channel
last_request_datetime = {}

if not os.path.exists('./tmp'):
	os.makedirs('./tmp')

def update_app_home(body, logger):
	user_id = body["event"]["user"]
	# Get the block kit structure from home.py
	blocks = build_app_home_blocks()
	try:
		slack_client.views_publish(
			user_id=user_id,
			view={
				"type": "home",
				"blocks": blocks
			}
		)
	except SlackApiError as e:
		logger.error(f"Error publishing App Home: {e}")

def update_home_tab(slack_client, event, logger):
	try:
		slack_client.views_publish(
			user_id=event["user"],
			view={
				"type": "home",
				"blocks": build_app_home_blocks(),
			},
		)
	except Exception as e:
		logger.error(f"Error publishing home tab: {e}")



@app.event("app_home_opened")
def handle_app_home(body, logger):
	user = body["event"]["user"]
	logger.info(f"App Home opened by user {user}")

	# Open a direct message channel with the user
	dm_channel = slack_client.conversations_open(users=user)

	# Get the channel ID from the response
	channel_id = dm_channel["channel"]["id"]
	dm_channel_ids[user] = channel_id

	# Fetch the conversation history using the channel ID
	result = slack_client.conversations_history(channel=channel_id, limit=1)

	if not result["messages"]:
		# Send a welcome message if no previous message exists
		welcome_message = "Welcome:robot_face:! To start interacting with me, send a message right here! To utilize DALL-E (a powerful image generation AI), start your input statement with image:\nGPT MODEL: gpt-3.5-turbo and 4"
		slack_client.chat_postMessage(channel=user, text=welcome_message)

	# Update the App Home tab
	update_app_home(body, logger)

@app.event("reaction_added")
def handle_reaction_added_events(body, logger):
  print('someone reacted')

#disabled
# @app.action("go_to_messages")
def handle_go_to_messages(ack, body, logger, slack_client):
	user_id = body["user"]["id"]
	ack()

	prompt = "puppy at sunset"  # Modify this as needed
	image_url = generate_image(prompt)

	slack_client.chat_postEphemeral(
		channel=body["container"]["channel_id"],
		user=user_id,
		text=f"Here is an image of a {prompt}: {image_url}\nTo generate more images, please send a message like `image: {prompt}` in the *Messages* tab of the app home."
	)


# Buttons to link out to a URL of policies and resources of your choice in Slack App Home. Build a App home with Slack's block kit builder or turn off this feature from slack app setup
@app.action("policies-button-action")
def handle_some_action(ack, body, logger):
		ack()
		logger.info(body)

@app.shortcut("chatgpt")
def handle_shortcuts(ack, body, logger):
		ack()
		logger.info(body)

@app.action("button-action")
def handle_some_action(ack, body, logger):
		ack()
		logger.info(body)

@app.event('app_mention')
def handle_app_mentions(body, logger):
	prompt = str(body['event']['text']).strip()
	channel = body['event']['channel']
	user = body['event']['user']
	thread_ts = body['event']['thread_ts'] if 'thread_ts' in body['event'] else None
	handle_prompt(prompt, channel, user, thread_ts, direct_message=True)


# Activated when the bot receives a direct message
@app.event('message')
def handle_message_events(body, logger):
	bot_id = body['event'].get('bot_id')
	# if body['event'].get('subtype') == 'bot_message' or (bot_id is not None and '@' + bot_id in body['event']['text']):
	# 	return  # Ignore bot messages and messages that tag the bot (handled by app_mention)
	if body['event'].get('subtype') == 'bot_message':
		return # Ignore bot messages

	prompt = str(body['event']['text']).strip()
	channel = body['event']['channel']
	user = body['event']['user']
	log(f'User {user} in Channel {channel} message: {prompt}')
	is_im_channel = slack_client.conversations_info(channel=channel)['channel']['is_im']

	if is_im_channel:
		thread_ts = body['event']['thread_ts'] if 'thread_ts' in body['event'] else None
		handle_prompt(prompt, channel, user, thread_ts, direct_message=True)


def handle_prompt(prompt, user, channel, thread_ts=None, direct_message=False, in_thread=False):
	# Initialize the last request datetime for this channel
	if channel not in last_request_datetime:
		last_request_datetime[channel] = datetime.fromtimestamp(0)

	# Let the user know that we are busy with the request if enough time has passed since last message
	if last_request_datetime[channel] + timedelta(seconds=history_expires_seconds) < datetime.now():
		slack_client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text=random.choice(robot_working_messages))

	# Set current timestamp
	last_request_datetime[channel] = datetime.now()

	# Read parent message content if called inside thread conversation
	parent_message_text = None
	if thread_ts and not direct_message:
		conversation = slack_client.conversations_replies(channel=channel, ts=thread_ts)
		if len(conversation['messages']) > 0 and valid_input(conversation['messages'][0]['text']):
			parent_message_text = conversation['messages'][0]['text']

	# Handle empty prompt
	if len(prompt.strip()) == 0 and parent_message_text is None:
		return

	# Clean the prompts
	prompt = re.sub(r'<[^>]*>', '', prompt, count=1)
	prompt = prompt.split('@', 1)[0].strip()

	# Generate DALL-E image command based on the image prompt
	if prompt.lower().startswith('image: '):
		slack_client.chat_postMessage(
			channel=channel,
			text=f"Ok, :saluting_face: generating image for your request..."
		)
		base_image_prompt = prompt[6:].strip()
		image_prompt = base_image_prompt

		# Append parent message text as prefix if exists
		if parent_message_text:
			image_prompt = f'{parent_message_text}. {image_prompt}'
			log('Using parent message inside thread')

		if len(image_prompt) == 0:
			text = 'Please check your input. To generate image use this format -> image: robot walking a dog'
		else:
			image_url = generate_image(
				prompt=image_prompt,
				size=image_size,
				channel=channel,
				thread_ts=thread_ts)
			image_path = None

			try:
				image_content = urlopen(image_url).read()

				short_prompt = base_image_prompt[:30] if validate_input_none_or_empty(base_image_prompt) else image_prompt[:30].strip(',/_')
				image_name = f"{short_prompt.replace(' ', '_')}.png"
				image_path = f'./tmp/{image_name}'

				image_file = open(image_path, 'wb')
				image_file.write(image_content)
				image_file.close()

				upload_response = slack_client.files_upload_v2(
					channel=dm_channel_ids.get(user, user),
					thread_ts=thread_ts,
					title=base_image_prompt,
					filename=image_name,
					file=image_path
				)

				text = upload_response['file']['url_private']
			except SlackApiError as e:
				text = None
				log(f'Slack API error: {e}', error=True)

			# uncomment to save storage space
		  # Remove temp image
			# if image_path and os.path.exists(image_path):
			#    os.remove(image_path)

	# Generate chat response
	else:
		now = datetime.now()
		# Create the file path
		file_path = './tmp/history_messages_day.txt'

		# Read the array of questions from the file
		if os.path.exists(file_path):
			with open(file_path, 'r') as file:
				history_messages = file.readlines()
		else:
			history_messages = []

		# Remove newline characters from each question
		history_messages = [question.strip() for question in history_messages]
		if channel in chat_history:
			for channel_message in chat_history[channel]:
				if channel_message['created_at'] + timedelta(seconds=history_expires_seconds) < now or \
						channel_message['thread_ts'] != thread_ts or parent_message_text == channel_message['content']:
					continue
				history_messages.append({'role': channel_message['role'], 'content': channel_message['content']})
		else:
			chat_history[channel] = []

		log(f'Using {len(history_messages)} messages from chat history')

		# Append parent text message from current thread
		if parent_message_text:
			history_messages.append({'role': 'user', 'content': parent_message_text})
			log(f'Adding parent message from thread with timestamp: {thread_ts}')

		# Combine messages from history, current prompt and system if not disabled
		messages = [
			*history_messages,
			{'role': 'user', 'content': prompt}
		]
		if system_desc.lower() != 'none':
			messages.insert(0, {'role': 'system', 'content': system_desc})

		#todo: clean
		try:
			response = openai_client.chat.completions.create(model=gptmodel, messages=messages)
		except Exception as e:
			log(f'ChatGPT response error: {e}', error=True)
			# Reply with error message
			slack_client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=str(e))
			return

		text = response.choices[0].message.content.strip('\n')

		chat_history[channel].append({'role': 'user', 'content': prompt, 'created_at': now, 'thread_ts': thread_ts})
		chat_history[channel].append(
			{'role': 'assistant', 'content': text, 'created_at': datetime.now(), 'thread_ts': thread_ts})

		# Remove the oldest 2 history message if the channel history size is exceeded for the current thread
		if len(list(filter(lambda x: x['thread_ts'] == thread_ts, chat_history[channel]))) >= (history_size + 1) * 2:
			# Create iterator for chat history list
			chat_history_list = (msg for msg in chat_history[channel] if msg['thread_ts'] == thread_ts)
			first_occurance = next(chat_history_list, None)
			second_occurance = next(chat_history_list, None)

			# Remove first occurance
			if first_occurance:
				chat_history[channel].remove(first_occurance)

			# Remove second occurance
			if second_occurance:
				chat_history[channel].remove(second_occurance)

		# Reply answer to thread
		if direct_message:
			target_channel = dm_channel_ids.get(user, user)
		else:
    			target_channel = channel

		slack_client.chat_postMessage(channel=target_channel, thread_ts=thread_ts, text=text, reply_broadcast=in_thread)

	slack_client.chat_postMessage(
			channel=channel,
			text=f"Done generating :D"
		)
	log(f'ChatGPT response: {text}')


if __name__ == '__main__':
	logging.basicConfig(level='INFO')

	print(f'ChatGPT Slackbot version 0.0.1a')
	handler = SocketModeHandler(app, SLACK_APP_TOKEN)
	handler.start()
