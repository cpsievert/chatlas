# chatlas

Easily chat with various LLM models from Ollama, Anthropic, OpenAI, and more.
`chatlas` is intentionally minimal -- making it easy to get started, while also supporting important features like streaming, tool calling, images, async, and more.

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```
pip install git+https://github.com/cpsievert/chatlas
```

After installing, you'll want to pick a [model provider](#model-providers), and get [credentials](#managing-credentials) set up (if necessary). Here, we demonstrate usage with OpenAI, but the concepts here apply to other implementations as well.


## Using chatlas

You can chat via `chatlas` in several different ways, depending on whether you are working interactively or programmatically. They all start with creating a new chat object:

```python
from chatlas import ChatOpenAI

chat = ChatOpenAI(
  model = "gpt-4o-mini",
  system_prompt = "You are a friendly but terse assistant.",
)
```

Chat objects are stateful: they retain the context of the conversation, so each new query can build on the previous ones. This is true regardless of which of the various ways of chatting you use.

### Interactive console

The most interactive, least programmatic way of using `chatlas` is to chat with it directly in your console with `chat.console()` or in your browser with `chat.app()`.

```python
chat.console()
```

```
Entering chat console. Press Ctrl+C to quit.

?> Who created Python?

Python was created by Guido van Rossum. He began development in the late 1980s and released the first     
version in 1991. 

?> Where did he develop it?

Guido van Rossum developed Python while working at Centrum Wiskunde & Informatica (CWI) in the            
Netherlands.     
```

The chat console is useful for quickly exploring the capabilities of the model, especially when you’ve customized the chat object with tool integrations (covered later).

The chat app is similar to the chat console, but it runs in your browser. It's useful if need more interactive capabilities like easy copy-paste.

```python
chat.app()
```

<div align="center">
<img width="600" alt="Using chatlas in a Shiny web app" src="https://github.com/user-attachments/assets/be4c1328-d7ff-49a9-9ac4-d1cf9a68b80d">
</div>


Again, keep in mind that the chat object retains state, so when you enter the chat console, any previous interactions with that chat object are still part of the conversation, and any interactions you have in the chat console will persist even after you exit back to the Python prompt.


### Interactive chat

The second most interactive way to chat is by calling the `chat()` method (from the normal Python prompt or in a script):

```python
chat.chat("What preceding languages most influenced Python?")
```

```
Python was primarily influenced by ABC, with additional inspiration from C,
Modula-3, and various other languages.
```

Since `chat()` is designed for interactive use, it prints the response to the console as it arrives rather than returning anything. This is useful when you want to display the response as it arrives, but don't need an interactive console. 

If you want to do something with the (last) response, you can use `last_turn()` or `turns()` to access conversation "turns". A turn contains various information about what happens during a "user" or "assistant" turn, but usually you'll just want the text of the response:

```python
chat.last_turn().text
```

```
"Python was primarily influenced by ABC, with additional inspiration from C, Modula-3, and various other languages."
```

### Programmatic Chat

For a more programming friendly interface, you can `submit()` a user turn and get a [generator](https://wiki.python.org/moin/Generators) of strings back. In the default case of `stream=True`, the generator yields strings as they arrive from the API (in small chunks). This is useful when you want to process the response as it arrives and/or when the response is too long to fit in memory.

```python
response = chat.submit("What is 1+1?", stream=True)
for x in response:
    print(x, end="")
```

```
1 + 1 equals 2.
```

With `stream=False` you still get a generator, but it yields the entire response at once. This is primarily useful as workaround: some models happen to not support certain features (like tools) when streaming. Also, more generally, sometimes it's useful to have response before displaying anything.

```python
response = chat.submit("What is 1+1?", stream=False)
for x in response:
    print(x)
```

```
1 + 1 equals 2.
```

### Vision (Image Input)

To ask questions about images, you can pass one or more additional input arguments using `content_image_file()` and/or `content_image_url()`:

```python
chat.chat(
    content_image_url("https://www.python.org/static/img/python-logo.png"),
    "Can you explain this logo?"
)
```

```
The Python logo features two intertwined snakes in yellow and blue,
representing the Python programming language. The design symbolizes...
```

The `content_image_url()` function takes a URL to an image file and sends that URL directly to the API. The `content_image_file()` function takes a path to a local image file and encodes it as a base64 string to send to the API. Note that by default, `content_image_file()` automatically resizes the image to fit within 512x512 pixels; set the `resize` parameter to "high" if higher resolution is needed.


## Model providers

`chatlas` supports various LLM models from Anthropic, OpenAI, Google, Ollama, and others.
Options like Anthropic, OpenAI, and Google require an account and API key to use, and also send your input to a remote server for response generation.
[Ollama](#ollama), on the other hand, provides a way to run open source models that run locally on your own machine, so is a good option for privacy and cost reasons.


### Anthropic

To use Anthropic's models (i.e., Claude), you'll need to sign up for an account and [get an API key](https://docs.anthropic.com/en/api/getting-started).
You'll also want the Python package:

```shell
pip install anthropic
```

Paste your API key into `ChatAnthropic()` to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import ChatAnthropic
chat = ChatAnthropic(api_key="...")
```


### OpenAI

To use OpenAI's models (i.e., GPT), you'll need to sign up for an account and [get an API key](https://platform.openai.com/docs/quickstart).
You'll also want the Python package:

```shell
pip install openai
```

Paste your API key into `ChatOpenAI()` to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import ChatOpenAI
chat = ChatOpenAI(api_key="...")
```


### Google

To use Google's models (i.e., Gemini), you'll need to sign up for an account and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python).
You'll also want the Python package:

```shell
pip install google-generativeai
```

Paste your API key into `ChatGoogle()` to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import ChatGoogle
chat = ChatGoogle(api_key="...")
```

### Ollama

To use Ollama, first download and run the [Ollama](https://ollama.com/) executable. Then [choose a model](https://ollama.com/library), like llama 3.2, to download (from the command line):

```shell
ollama run llama-3.2
```

You'll also want the Python package:

```shell
pip install ollama
```

Now, you're read to chat via `chatlas`:

```python
from chatlas import ChatOllama
chat = ChatOllama(model="llama3.2")
```

## Groq

To use [Groq](https://groq.dev/), you'll need to obtain an API key. You'll also want the `openai` Python package:

```shell
pip install openai
```

Paste your API key into `ChatGroq()` to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import ChatGroq
chat = ChatGroq(api_key="...")
```

### AWS Bedrock

[AWS Bedrock](https://aws.amazon.com/bedrock/) provides a number of chat based models, including those Anthropic's [Claude](https://aws.amazon.com/bedrock/claude/). To use AWS Bedrock, you'll need the `anthropic` Python package, along with `bedrock` extras:

```python
pip install anthropic[bedrock]
```

Then, give your AWS deployment to `ChatBedrockAnthropic()`. Alternatively, see [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) for a more detailed explanation of how to properly manage your AWS credentials.

```python
from chatlas import ChatBedrockAnthropic

chat = ChatBedrockAnthropic(
  aws_profile='...',
  aws_region='us-east'
  aws_secret_key='...',
  aws_access_key='...',
  aws_session_token='...',
)
```


### Azure

To use [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service), you'll need the `openai` Python package:

```shell
pip install openai
```

Then, pass along information about your Azure deployment to the `ChatAzureOpenAI` constructor:

```python
import os
from chatlas import ChatAzureOpenAI

chat = ChatAzureOpenAI(
  endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  deployment_id='REPLACE_WITH_YOUR_DEPLOYMENT_ID',
  api_version="YYYY-MM-DD",
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
)
```


```

<!--

### LangChain

To use LangChain's [chat models](https://python.langchain.com/docs/integrations/chat/), you'll need to follow the relevant setup instructions ([for example](https://python.langchain.com/docs/integrations/chat/openai/#setup))
You'll also want the relevant Python packages:

```shell
pip install langchain langchain-openai
```

Then, once you have a chat model instance, pass it to the `LangChainChat` constructor:

```python
from chatlas import LangChainChat
from langchain_openai import ChatOpenAI

chat = LangChainChat(ChatOpenAI())
```
-->


## Managing credentials

Pasting an API key into a chat constructor (e.g., `ChatOpenAI(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.
Instead, consider using environment variables or a configuration file to manage your credentials.
One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

```shell
pip install python-dotenv
```

```shell
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

```python
from chatlas import Anthropic
from dotenv import load_dotenv

load_dotenv()
chat = ChatAnthropic()
chat.console()
```

Another option, which makes interactive use easier, is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

```shell
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

## Advanced features

### Tool (function) calling

Tool calling is a powerful feature enabling the LLM to call external programs to help answer your questions.
`chatlas` makes it easy to provide Python functions as tools for the LLM to call, and handles the communication between the LLM and your function.

To provide a tool, just define function(s) and pass them to the `chatlas` constructor.
Make sure to annotate your function with types to help the LLM understand what it should expect and return.
Also provide a docstring to help the LLM understand what your function does.

```python
from chatlas import ChatAnthropic

def get_current_weather(location: str, unit: str = "fahrenheit") -> int:
    """Get the current weather in a location."""
    if "boston" in location.lower():
        return 12 if unit == "fahrenheit" else -11
    elif "new york" in location.lower():
        return 20 if unit == "fahrenheit" else -6
    else:
        return 72 if unit == "fahrenheit" else 22

chat = ChatAnthropic()
chat.register_tool(get_current_weather)
chat.chat("What's the weather like in Boston, New York, and London today?")
```

### Data extraction

To extract structured data you call the `.extract_data()` method instead of the `.chat()` method.

To extract data, you need to define a function that takes the LLM's response as input and returns the extracted data. You’ll also need to define a [pydantic model](https://docs.pydantic.dev/latest/#why-use-pydantic) that describes the structure of the data that you want. Here’s a simple example that extracts two specific values from a string:

```python
from chatlas import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    age: int
    name: str

chat = ChatOpenAI()
chat.extract_data("My name is Susan and I'm 13 years old", data_model=Person)
```

```
{'age': 13, 'name': 'Susan'}
```


### Build your own Shiny app

Pass user input from a Shiny `Chat()` component to a `chatlas` response generator to embed a chat interface in your own Shiny app.

```python
from chatlas import ChatAnthropic
from shiny import ui

chat = ui.Chat(
  id="chat", 
  messages=["Hi! How can I help you today?"],
)

llm = ChatAnthropic()

@chat.on_user_submit
def _(message):
    response = llm.submit(message)
    chat.append_message_stream(response)
```