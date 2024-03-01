# OpenAI ChatGPT Slackbot

A powerful Slackbot codebase built by High-Tower80 with OpenAI's GPT capabilities. This app allows users to interact with GPT-4 directly from Slack. It also supports generating images with DALL-E by giving a text prompt and saves it to the slackbot's DM. It also supports text-to-speech and many more updates. This fork will require a lot of revamps and refactors. I do this on my spare time.

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20image2.jpeg)

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/Slackgpt%20summary.png)

## Features

1. Directly chat with GPT-4 from your Slack workspace.
2. Generate DALL-E images with a text prompt.
3. Automatically update the App Home tab with a block kit structure.
4. Handle bot mentions and direct messages.
5. Maintain chat history to provide context for future prompts.

## Getting Started

Before you can use the OpenAI ChatGPT Slackbot, you need to set up and configure your environment. This includes Slack bot setup and a lot of toggling.

### Prerequisites

1. Python 3.11+
2. Slack `signing_secret`
3. Slack `app_level_token`
4. Slack `user_oauth_token`
5. OpenAI `open_api_key`

### Slack bot and App scopes

```json
{
  "scopes": {
    "bot": [
      "app_mentions:read",
      "channels:history",
      "channels:join",
      "chat:write",
      "commands",
      "files:write",
      "groups:history",
      "im:history",
      "im:read",
      "im:write",
      "incoming-webhook",
      "mpim:history",
      "users:read",
      "workflow.steps:execute",
      "files:read"
    ]
  }
}
```

### Setup

Intermidiate developer skill highly recommended to continue.

1. Install the required packages

```
pip install -r requirements.txt
```

2. Set up the Slack bot following the instructions [here](https://api.slack.com/start).
   - Slack channel/workspace setup
   - Slack bot setup
3. openAI setup

   - if you need to use Dall-e then you need a payment method attached to your openAI account

4. Set up your environment variables in a `.env` file in root folder
   - see `.env.example`

### Running the Slackbot

Run the Slackbot with the following command:

```
python main.py
```

## Usage

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20sheets.png)

### Generating Images with DALL-E

To generate an image with DALL-E, send a message in the following format:

```
@bot image: Your image description here
```

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20image1.jpeg)

## Disclaimer

- This project is for demonstration purposes only and is not officially associated with OpenAI.
- This project does not include error handling for situations where the API keys are not set or incorrect. Please ensure that your API keys are correct before running the bot.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
