import os
import logging

from utils.api.slack import SlackBot

if __name__ == '__main__':
	logging.basicConfig(level='INFO')

	if not os.path.exists('./tmp'):
		os.makedirs('./tmp')

	bot = SlackBot()
	bot.start_bot()
