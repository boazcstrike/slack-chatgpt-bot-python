from openai import OpenAI
from dotenv import load_dotenv

from utils.core import get_env, log

class OpenAIAPI():
  """
  OpenAPI Documentation
  https://platform.openai.com/docs/api-reference

  Examples:
  https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
  """
  def __init__(self, token):
    load_dotenv()
    self.token = token
    self.client = OpenAI(
      api_key = get_env('OPENAI_API_KEY', None),
    )
    self.gptmodel = get_env('GPT_MODEL', 'gpt-4')
    self.system_desc = get_env('GPT_SYSTEM_DESC', 'Helpful AI assistant.')
    self.file_path = './tmp/prompts_history.txt'

    self.size_map = {
      'dall-e-2': [
        '256x256',
        '512x512',
        '1024x1024',
      ],
      'dall-e-3': [
        '1024x1792',
        '1792x1024',
        '1024x1024',
      ]
    }
    self.slack = None

    self.settings = {
       "history_expires_seconds": get_env('HISTORY_EXPIRES_IN', '900'),
       "history_size": get_env('HISTORY_SIZE', '10'),
    }

  def create_completion(self, prompt, max_tokens=None, temperature=1, top_p=1.0, n=1, stop=None, timeout=None, presence_penalty=0, frequency_penalty=0, best_of=1, stream=False, logprobs=None, logit_bias=None):
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """
    response = self.client.chat.completions.create(
      model=self.gptmodel,
      frequency_penalty=frequency_penalty,
      logit_bias=logit_bias,
      logprobs=logprobs,
      max_tokens=max_tokens,
      n=n,
      presence_penalty=presence_penalty,
      stop=stop,
      stream=stream,
      prompt=prompt,
      temperature=temperature,
      top_p=top_p,
      timeout=timeout,
      best_of=best_of,
    )
    return response

  def validate_image_size(self, size, model):
    """
    Validates the image size for the given model.
    """
    if size not in self.size_map.get(model, []):
      raise ValueError(f"Invalid size '{size}' for model '{model}'")

  def generate_image(
      self,
      prompt,
      size,
      quality="standard",
      model="dall-e-3",
      slack=None):
    """
    generates an image from a prompt using the DALL-E API.

    full doc: https://platform.openai.com/docs/api-reference/images/create
    """
    log(f'Attempting to generate "{prompt}" at {quality} {size}')
    self.validate_image_size(size, model)

    try:
      response = self.client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1
      )
      image_url = response.data[0].url
    except Exception as e:
      log(f'ChatGPT response error: {e}', error=True)
      if slack:
        slack.chat(text=str(e))
      return
    return image_url