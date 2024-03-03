import os
import logging
import sys

from utils.api.slack import SlackBot
from utils.vars.main import required_env

if __name__ == '__main__':
    # for e in required_env:
    #     if os.environ.get(e) is None:
    #         print("Environment variable", e, "required but not set")
    #         sys.exit(1)

    logging.basicConfig(level='INFO')
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    bot = SlackBot()
    bot.start_bot()
