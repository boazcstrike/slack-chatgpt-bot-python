import os
from dotenv import load_dotenv
from openai import OpenAI
from urllib.request import urlopen

load_dotenv()
c = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY'),
)

prompt = input('Enter a prompt: ')

print(f"generating...")
image_response = c.images.generate(
    model='dall-e-3',
    prompt=prompt,
    size='1024x1024',
    quality='standard',
    n=1,
)
print('\nimage_response', image_response)

image_url = image_response.data[0].url
print('\nimage_url', image_url)
image_content = urlopen(image_url).read()

image_name = f"{prompt.replace(' ', '_')}.png"
image_path = f'./tmp/{image_name}'
image_file = open(image_path, 'wb')
image_file.write(image_content)
image_file.close()
print(f'\nsaved image to {image_path}')
