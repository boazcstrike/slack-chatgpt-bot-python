import tiktoken


encoding = tiktoken.get_encoding("cl100k_base")

# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4")
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

num_tokens = len(encoding.encode('Hi, how are you?'))
print(num_tokens)
