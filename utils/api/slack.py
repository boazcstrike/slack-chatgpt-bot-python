import os
import random
import re

from dotenv import load_dotenv
from datetime import datetime, timedelta
from urllib.request import urlopen

from slack_bolt import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from utils.core import get_env, log, validate_input
from utils.home import build_app_home_blocks
from utils.api.openai import OpenAIAPI


bot_busy_messages = [
  'Generating... :gear:',
  'Beep beep :robot_face:',
  'hm :thinking_face:',
  'On it :saluting_face:',
  'Processing... :hourglass:',
  'Analyzing data... :bar_chart:',
  'Creating magic... :sparkles:',
  'Thinking hard... :thought_balloon:',
  'In progress... :construction:',
  'Brainstorming... :cloud:',
  'Almost there... :soon:',
  'Calculating... :abacus:',
  'Inspiration struck... :bulb:',
  'Stay tuned... :tv:',
  'Hold tight... :seat:',
  'Getting creative... :art:',
  'Time to shine... :star2:',
  'Data crunching... :chart_with_downwards_trend:',
  'Awaiting command... :control_knobs:',
  'Plotting course... :compass:',
  'Working its magic... :sparkler:',
  'Pondering... :thinking_face:',
  'On the case... :mag_right:',
  'Analyzing patterns... :spider_web:',
  'Deep in thought... :thinking:',
  'Progressing steadily... :runner:',
  'Processing data... :floppy_disk:',
  'Initiating process... :bulb:',
  'Let me think... :thinking_face:',
  'Almost done... :soon:',
  'Crunching numbers... :1234:',
  'Plotting strategy... :chart_with_upwards_trend:',
  'Stay patient... :hourglass_flowing_sand:',
  'Generating insights... :bulb:',
  'Charging ahead... :rocket:',
  'Searching for answers... :mag:',
  'Unlocking secrets... :key:',
  'Configuring settings... :gear:',
  'Constructing solutions... :hammer:',
  'Solving mysteries... :detective:',
  'Shaping ideas... :bulb:',
  'Formulating plan... :memo:',
  'Building bridges... :bridge_at_night:',
  'Firing up engines... :fire:',
  'Unleashing power... :zap:',
  'Mapping pathways... :world_map:',
  'Sifting through data... :bar_chart:',
  'Engaging gears... :gear:',
  'Generating results... :chart_with_upwards_trend:',
  'Designing solutions... :bulb:',
  'Investigating options... :mag:',
  'Assembling components... :hammer_and_wrench:',
  'Refining processes... :gear:',
  'Deciphering codes... :key:',
  'Processing information... :floppy_disk:',
  'Crafting masterpiece... :art:',
  'Weaving magic... :sparkles:',
  'Navigating possibilities... :compass:',
  'Exploring horizons... :telescope:',
  'Breaking ground... :construction:',
  'Building foundations... :bricks:',
  'Aligning stars... :stars:',
  'Executing plans... :dart:',
  'Stirring creativity... :artist:',
  'Evoking brilliance... :bulb:',
  'Initiating sequence... :1234:',
  'Activating potential... :sparkler:',
  'Igniting innovation... :fire:',
  'Exploring realms... :world_map:',
  'Venturing forward... :mountain:',
  'Delving deeper... :mag:',
  'Crafting solutions... :wrench:',
  'Generating ideas... :bulb:',
  'Analyzing possibilities... :bar_chart:',
  'Synthesizing data... :test_tube:',
  'Plotting trajectory... :rocket:',
  'Optimizing pathways... :chart_with_upwards_trend:',
  'Seeking insights... :mag:',
  'Forging ahead... :mountain:',
  'Unlocking potential... :key:',
  'Formulating strategies... :clipboard:',
  'Envisioning success... :crystal_ball:',
  'Creating pathways... :construction:',
  'Navigating complexities... :compass:',
  'Evaluating options... :chart_with_downwards_trend:',
  'Building momentum... :rocket:',
  'Charting course... :world_map:',
  'Harnessing creativity... :art:',
  'Analyzing trends... :chart_with_upwards_trend:',
  'Exploring avenues... :mag:',
  'Refining concepts... :bulb:',
  'Activating solutions... :gear:',
  'Initiating processes... :hourglass:',
  'Unlocking mysteries... :key:',
  'Engineering innovation... :gear:',
  'Exploring frontiers... :telescope:',
  'Optimizing strategies... :dart:',
  'Constructing dreams... :hammer_and_wrench:',
  'Inspiring greatness... :sparkles:',
  'Calculating outcomes... :1234:',
  'Generating possibilities... :bulb:',
  'Pioneering pathways... :compass:',
  'Solving puzzles... :puzzle_piece:',
  'Nurturing ideas... :seedling:',
  'Igniting creativity... :fire:',
  'Weaving stories... :scroll:',
  'Navigating challenges... :ship:',
  'Architecting solutions... :building_construction:',
  'Harvesting insights... :bulb:',
  'Chasing dreams... :runner:',
  'Analyzing results... :chart_with_upwards_trend:',
  'Transforming ideas... :gear:',
  'Unleashing potential... :lightning:',
  'Inspiring imagination... :crystal_ball:',
  'Building legacies... :bricks:',
  'Empowering innovation... :rocket:',
  'Exploring possibilities... :mag:',
  'Activating creativity... :paintbrush:',
  'Charting progress... :chart_with_upwards_trend:',
  'Crafting strategies... :hammer_and_wrench:',
  'Shaping futures... :world_map:',
  'Engaging imagination... :thought_balloon:',
  'Breaking boundaries... :chains:',
  'Forging pathways... :hammer_and_pick:',
  'Pioneering solutions... :compass:',
  'Sparking ideas... :bulb:',
  'Initiating action... :rocket:',
  'Navigating waters... :sailboat:',
  'Crafting visions... :crystal_ball:',
  'Building bridges... :bridge_at_night:',
  'Shaping worlds... :globe_with_meridians:',
  'Harnessing potential... :zap:',
  'Igniting passion... :fire:',
  'Exploring terrain... :compass:',
  'Fueling innovation... :rocket:',
  'Solving challenges... :dart:',
  'Sculpting success... :art:',
  'Constructing realities... :building_construction:',
  'Charting directions... :world_map:',
  'Activating imagination... :bulb:',
  'Pioneering frontiers... :mountain:',
  'Guiding journeys... :compass:',
  'Empowering change... :zap:',
  'Harvesting creativity... :seedling:',
  'Engineering progress... :hammer_and_wrench:',
  'Dreaming big... :mountain:',
  'Exploring depths... :diving_mask:',
  'Forging destinies... :chains:',
  'Breaking barriers... :hammer:',
  'Crafting narratives... :scroll:',
  'Innovating futures... :rocket:',
  'Navigating stars... :telescope:',
  'Shaping destinies... :hammer_and_pick:',
  'Championing innovation... :trophy:',
  'Building empires... :building_construction:',
  'Igniting revolutions... :fire:',
  'Exploring galaxies... :milky_way:',
  'Forging alliances... :handshake:',
  'Empowering creativity... :art:',
  'Harvesting potentials... :seedling:',
  'Guiding vision... :compass:',
  'Navigating paths... :world_map:',
  'Engineering tomorrow... :rocket:',
  'Chasing horizons... :sunrise:',
  'Building foundations... :construction:',
  'Dreaming grand... :sparkles:',
  'Pioneering voyages... :sailboat:',
  'Cultivating dreams... :seedling:',
  'Forging opportunities... :hammer_and_anvil:',
  'Sculpting dreams... :sculpture:',
  'Innovating tomorrows... :rocket:',
  'Shaping destinies... :world_map:',
  'Igniting revolutions... :fireworks:',
  'Exploring frontiers... :mountain:',
  'Forging alliances... :handshake:',
  'Empowering creativity... :art:',
  'Harvesting potentials... :seedling:',
  'Guiding vision... :compass:',
  'Navigating paths... :world_map:',
  'Engineering tomorrow... :rocket:',
  'Chasing horizons... :sunrise:',
  'Building foundations... :construction:',
  'Dreaming grand... :sparkles:',
  'Pioneering voyages... :sailboat:',
  'Cultivating dreams... :seedling:',
  'Forging opportunities... :hammer_and_anvil:',
  'Sculpting dreams... :sculpture:',
  'Innovating tomorrows... :rocket:',
  'Shaping destinies... :world_map:',
  'Igniting revolutions... :fireworks:',
  'Exploring frontiers... :mountain:',
]


class SlackBot():
  def __init__(self):
    print('starting slack bot connection...')
    load_dotenv()
    self.token = os.getenv('SLACK_BOT_USER_OAUTH_TOKEN', None)
    self.signing_secret = os.getenv('SLACK_BOT_SIGNING_SECRET', None)
    self.client = WebClient(self.token)
    self.bot_id = self.client.auth_test().data['user_id']
    self.bot_name = self.client.auth_test().data['user']
    self.bot_channel = 'ai-dump'
    self.dm_channel_ids = {}
    self.last_request_datetime = {}
    self.busy_messages = bot_busy_messages

    self.app = App(
      token=self.token,
      signing_secret=self.signing_secret,
    )
    self.register_listeners()

  def start_bot(self):
    print(f'\n\n\033[1mStarted Slackbot Version 0.1.0b\033[0m\n\n')
    self.app.start()

  def send_message(self, channel, message, blocks=None, thread_ts=None, reply_broadcast=None):
    post_message = {
      "channel": channel,
      "text": message,
      "blocks": blocks,
      "thread_ts": thread_ts,
      "reply_broadcast": reply_broadcast,
    }

    if blocks:
      post_message.update({"blocks": blocks})
    if thread_ts:
      post_message.update({"thread_ts": thread_ts})
    if reply_broadcast:
      post_message.update({"reply_broadcast": reply_broadcast})

    self.client.chat_postMessage(post_message)

  def send_message_with_attachment(self, channel, message, attachment):
    self.client.chat_postMessage(
      channel=channel,
      text=message,
      attachments=attachment
    )

  def send_ephemeral_message(self, channel, user, message, blocks=None):
      if blocks:
          self.client.chat_postEphemeral(
              channel=channel,
              user=user,
              text=message,
              blocks=blocks
          )
      else:
          self.client.chat_postEphemeral(
              channel=channel,
              user=user,
              text=message
          )

  def send_ephemeral_message_with_attachment(self, channel, user, message, attachment):
      self.client.chat_postEphemeral(
          channel=channel,
          user=user,
          text=message,
          attachments=attachment
      )

  def send_ephemeral_message_with_blocks(self, channel, user, blocks):
      self.client.chat_postEphemeral(
          channel=channel,
          user=user,
          blocks=blocks
      )

  def send_attachment(self, channel, attachment):
      self.client.chat_postMessage(
          channel=channel,
          attachments=attachment
      )

  def send_attachment_with_blocks(self, channel, message, attachment, blocks):
      self.client.chat_postMessage(
          channel=channel,
          text=message,
          attachments=attachment,
          blocks=blocks
      )

  def send_attachment_with_ephemeral(self, channel, user, attachment):
      self.client.chat_postEphemeral(
          channel=channel,
          user=user,
          attachments=attachment
      )

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
    """! does not work"""
    print('registering listeners...')
    @self.app.event("app_mention")
    def handle_app_mentions(body, say, payload):
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

  def handle_msg(self, message, user, channel, thread_ts=None, direct_message=False, in_thread=False):
    if channel not in self.last_request_datetime:
      self.last_request_datetime[channel] = datetime.fromtimestamp(0)
    # Let the user know that we are busy with the request if enough time has passed since last message
    if self.last_request_datetime[channel] + timedelta(seconds=self.history_expires_seconds) < datetime.now():
      self.client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=random.choice(self.busy_messages))

    self.last_request_datetime[channel] = datetime.now()

    # Read parent message content if called inside thread conversation
    parent_message_text = None
    if thread_ts and not direct_message:
      conversation = self.client.conversations_replies(
         channel=channel, ts=thread_ts)
      if len(conversation['messages']) > 0 and validate_input(conversation['messages'][0]['text']):
        parent_message_text = conversation['messages'][0]['text']

    prompt = re.sub(r'<[^>]*>', '', prompt, count=1)
    prompt = prompt.split('@', 1)[0].strip()

    if len(prompt.strip()) == 0 and parent_message_text is None:
      return
    if not prompt:
      raise ValueError("Prompt cannot be empty")

    if message.lower().startswith('image: '):
      self.handle_image_generation_prompt(
        channel,
        user,
        message,
        parent_message_text,
        thread_ts,
      )

    else:
      self.handle_chat_prompt(
        channel,
        user,
        message,
        parent_message_text,
        thread_ts,
        direct_message,
        in_thread
      )

  def _clean_image_prompt_message(self, prompt_message):
    """
    Cleans the image prompt message by removing the image: prefix and any leading/trailing whitespace.
    """
    return prompt_message[6:].strip(',/_')

  def handle_image_generation_prompt(
    self, channel, user,
    message, parent_message_text, thread_ts=None):
    self.send_message(
      channel=channel,
      message=f"Ok, :saluting_face: generating image for your request..."
    )
    prompt = self._clean_image_prompt_message(message, parent_message_text)

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
      size="1024x1024",
      slack=self)
    image_path = None

    try:
      image_content = urlopen(image_url).read()

      short_prompt = prompt[:50]
      image_name = f"{short_prompt.replace(' ', '_')}.png"
      image_path = f'./tmp/{image_name}'

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
        message=upload_response['file']['url_private'])
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

  def _save_chat_history(self, channel, message, parent_message_text, thread_ts):
    """this is for context windowing, to keep track of the last few messages in a channel for the chatbot to use as context."""
    file_path = f'./tmp/{channel}.txt'

    if os.path.exists(file_path):
      with open(file_path, 'r') as file:
        existing_messages = file.read().strip().split('\n')
    else:
      existing_messages = []

    for msg in existing_messages:
      existing_messages.append({'role': msg['role'], 'content': msg['content']})

    log(f'Using {len(existing_messages)} messages from chat history')

    if parent_message_text:
      existing_messages.append({'role': 'user', 'content': parent_message_text})
      log(f'Adding parent message from thread with timestamp: {thread_ts}')

    messages = [
      *existing_messages,
      {'role': 'user', 'content': message}
    ]
    messages.insert(0, {'role': 'system', 'content': 'You are a helpful assistant.'})

    with open(file_path, 'w') as file:
      file.write('\n'.join(messages))

    return messages

  def handle_size(self, history_size, messages, thread_ts=None):
    """Remove the oldest 2 history message if the channel history size is exceeded for the current threa"""
    if len(list(filter(lambda x: x['thread_ts'] == thread_ts, messages))) >= (history_size + 1) * 2:
      # Create iterator for chat history list
      chat_history_list = (msg for msg in messages if msg['thread_ts'] == thread_ts)
      first_occurance = next(chat_history_list, None)
      second_occurance = next(chat_history_list, None)

      # Remove first occurance
      if first_occurance:
        messages.remove(first_occurance)

      # Remove second occurance
      if second_occurance:
        messages.remove(second_occurance)


  def handle_chat_prompt(
    self, channel, user, message, parent_message_text, thread_ts=None, direct_message=False, in_thread=False):
    messages = self._save_chat_history(channel, message, parent_message_text, thread_ts)

    oa = OpenAIAPI()
    now = datetime.now()
    try:
      response = oa.chat.completions.create(messages=messages)
    except Exception as e:
      log(f'ChatGPT response error: {e}', error=True)
      self.send_message(channel=channel, thread_ts=thread_ts, text=str(e))
      return

    text = response.choices[0].message.content.strip('\n')

    messages.append({'role': 'user', 'content': message, 'created_at': now, 'thread_ts': thread_ts})
    messages.append(
      {'role': 'assistant', 'content': text, 'created_at': now, 'thread_ts': thread_ts})

    # todo: handle size so we can save some storage space and reset the contexts
    # self.handle_size(x, messages, thread_ts)

    if direct_message:
      target_channel = self.dm_channel_ids.get(user, user)
    else:
      target_channel = channel

    self.send_message(channel=target_channel, thread_ts=thread_ts, text=text, reply_broadcast=in_thread)

    log(f'ChatGPT response: {text}')
