import os
import logging

from slack_bolt import App
from utils.core import log

app = App(
	token=os.getenv('SLACK_BOT_USER_OAUTH_TOKEN', None),
	signing_secret=os.getenv('SLACK_BOT_SIGNING_SECRET', None),
)

@app.event("app_mention")
def handle_app_mentions(body, say, payload):
	try:
		message = str(body['event']['text']).strip()
		channel = body['event']['channel']
		user = body['event']['user']
		thread_ts = body['event']['thread_ts'] if 'thread_ts' in body['event'] else None
		bot.handle_msg(message, channel, user, thread_ts, direct_message=True)
	except Exception as e:
		say(f"Error handling app mention: {e}")
		log(f'Error handling app mention: {e}', error=True)

@app.event("reaction_added")
def handle_reaction_added_events(body, logger):
	print('someone reacted')


if __name__ == '__main__':
	logging.basicConfig(level='INFO')

	if not os.path.exists('./tmp'):
		os.makedirs('./tmp')

	app.start()
