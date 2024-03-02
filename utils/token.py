import tiktoken

class TokenCount():
    def __init__(self, token, count):
        self.token = token
        self.count = count

    def __str__(self):
        return f'{self.token} {self.count}'

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
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
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
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return num_tokens_from_messages(messages, model="gpt-4-0613")
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