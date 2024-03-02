import datetime

from openai import OpenAI
from dotenv import load_dotenv

from utils.core import get_env, log

class OpenAIAPI():
    """
    OpenAPI Documentation
    https://platform.openai.com/docs/api-reference

    Examples:
    https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models

    GPT Models:
    https://platform.openai.com/docs/models
    https://platform.openai.com/docs/guides/text-generation
    """
    def __init__(self):
        load_dotenv()
        self.token = get_env('OPENAI_API_KEY', None)
        self.client = OpenAI(
            api_key = self.token,
        )
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

    def create_completion(self, messages, max_tokens=None, temperature=1, top_p=1.0, n=1, stop=None, presence_penalty=0, frequency_penalty=0, stream=False, logprobs=None, logit_bias=None, model='gpt-3.5-turbo'):
        """
        https://platform.openai.com/docs/api-reference/chat/create
        https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        https://platform.openai.com/docs/models/gpt-3-5-turbo

        recommended models (i strongly recommend reading the official doc^ and browsing through the models)
        gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo

        gpt-3.5-turbo-0125
        gpt-4-1106-preview
        gpt-4-vision-preview

        legacy models:
        gpt-3.5-turbo-instruct
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
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
        size="1024x1024",
        quality="standard",
        model="dall-e-2",
        slack=None):
        """
        generates an image from a prompt using the DALL-E API.

        full doc: https://platform.openai.com/docs/api-reference/images/create
        """
        log(f'Attempting to generate "{prompt}" at {quality} {size} using {model}')
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
                return slack.send_message(message=str(e))
        return image_url

    def generate_image_variation(self, image, size="1024x1024", quality="standard", model="dall-e-2", slack=None):
        """generates image variations from a prompt and reference image using the DALL-E API."""
        log(f'Attempting to generate a variation at {quality} {size} using {model}')
        self.validate_image_size(size, model)

        try:
            response = self.client.images.create_variation(
            image=open(image, "rb"),
            n=2,
            size=size,
            )
            image_url = response.data[0].url
        except Exception as e:
            log(f'ChatGPT response error: {e}', error=True)
            if slack:
                return slack.send_message(message=str(e))

        return image_url


    def generate_tts(self, prompt, user, thread_ts=None):
        """generates a TTS audio file from a prompt"""
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%m-%d-%y-%H-%M-") + str(current_datetime.second)
        file_name = f"{formatted_datetime}-tts.mp3"
        speech_file_path = f"./tmp/"+file_name
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=prompt
        )
        with open(speech_file_path, 'wb') as f:
            response.stream_to_file(speech_file_path)

        return speech_file_path, file_name
