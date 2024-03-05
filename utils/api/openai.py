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
        self.default_chat_model = 'gpt-3.5-turbo'
        self.default_image_model = 'dall-e-2'

        self.settings = {
            "history_expires_seconds": get_env('HISTORY_EXPIRES_IN', '900'),
            "history_size": get_env('HISTORY_SIZE', '10'),
        }

    def create_completion(self, messages, max_tokens=None, temperature=1, top_p=1.0, n=1, stop=None, presence_penalty=0, frequency_penalty=0, stream=False, logprobs=None, logit_bias=None, model='gpt-4-turbo-preview'):
        """
        https://platform.openai.com/docs/api-reference/chat/create
        https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        https://platform.openai.com/docs/models/gpt-3-5-turbo

        recommended models (i strongly recommend reading the official doc^ and browsing through the models)
        gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo

        refer to def num_tokens_from_messages pls

        legacy models:
        gpt-3.5-turbo-instruct

        https://platform.openai.com/docs/api-reference/chat/object
        response example:
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                "role": "assistant",
                "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
        https://cookbook.openai.com/articles/techniques_to_improve_reliability
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
            self.clean_error_message(str(e))
            if slack:
                return slack._send_message(message=str(e))
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
            se_msg = self.clean_error_message(str(e))
            if slack:
                return slack._send_message(message=str(se_msg))

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

    def clean_error_message(error_message):
        error_json_start_index = error_message.find('{')
        error_json_end_index = error_message.rfind('}') + 1
        error_json = error_message[error_json_start_index:error_json_end_index]

        error_data = json.loads(error_json)

        error_code = error_data['error']['code']
        error_message = error_data['error']['message']

        return f'Error code: {error_code}'\
            f'Error message: {error_message}'

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        Returns the number of tokens in a text string.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def compare_encodings(example_string: str) -> None:
        """Prints a comparison of three string encodings."""
        # print the example string
        print(f'\nExample string: "{example_string}"')
        # for each encoding, print the # of tokens, the token integers, and the token bytes
        for encoding_name in ["r50k_base", "p50k_base", "cl100k_base"]:
            encoding = tiktoken.get_encoding(encoding_name)
            token_integers = encoding.encode(example_string)
            num_tokens = len(token_integers)
            token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
            print()
            print(f"{encoding_name}: {num_tokens} tokens")
            print(f"token integers: {token_integers}")
            print(f"token bytes: {token_bytes}")

    @staticmethod
    def count_tokens_from_message(messages, model="gpt-3.5-turbo"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return TokenCount.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return TokenCount.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens