# chatlas

<p>
<!-- badges start -->
<a href="https://pypi.org/project/chatlas/"><img alt="PyPI" src="https://img.shields.io/pypi/v/chatlas?logo=python&logoColor=white&color=orange"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://github.com/posit-dev/chatlas"><img src="https://github.com/posit-dev/chatlas/actions/workflows/test.yml/badge.svg?branch=main" alt="Python Tests"></a>
<!-- badges end -->
</p>

chatlas provides a simple and unified interface across large language model (llm) providers in Python. 
It abstracts away complexity from common tasks like streaming chat interfaces, tool calling, structured output, and much more.
chatlas helps you prototype faster without painting you into a corner; for example, switching providers is as easy as changing one line of code, but provider specific features are still accessible when needed.
Developer experience is also a key focus of chatlas: typing support, rich console output, and built-in tooling are all included.

(Looking for something similar to chatlas, but in R? Check out [elmer](https://elmer.tidyverse.org/)!)

## Install

Install the latest stable release from PyPI:

```bash
pip install -U chatlas
```

Or, install the latest development version from GitHub:

```bash
pip install -U git+https://github.com/posit-dev/chatlas
```

## Model providers

`chatlas` supports a variety of model providers. See the [API reference](https://posit-dev.github.io/chatlas/reference/index.html) for more details (like managing credentials) on each provider.

* Anthropic (Claude): [`ChatAnthropic()`](https://posit-dev.github.io/chatlas/reference/ChatAnthropic.html).
* GitHub model marketplace: [`ChatGithub()`](https://posit-dev.github.io/chatlas/reference/ChatGithub.html).
* Google (Gemini): [`ChatGoogle()`](https://posit-dev.github.io/chatlas/reference/ChatGoogle.html).
* Groq: [`ChatGroq()`](https://posit-dev.github.io/chatlas/reference/ChatGroq.html).
* Ollama local models: [`ChatOllama()`](https://posit-dev.github.io/chatlas/reference/ChatOllama.html).
* OpenAI: [`ChatOpenAI()`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html).
* perplexity.ai: [`ChatPerplexity()`](https://posit-dev.github.io/chatlas/reference/ChatPerplexity.html).
* vLLM: [`ChatVLLM()`](https://posit-dev.github.io/chatlas/reference/ChatVLLM.html).

It also supports the following enterprise cloud providers:

* AWS Bedrock: [`ChatBedrockAnthropic()`](https://posit-dev.github.io/chatlas/reference/ChatBedrockAnthropic.html).
* Azure OpenAI: [`ChatAzureOpenAI()`](https://posit-dev.github.io/chatlas/reference/ChatAzureOpenAI.html).

To use a model provider that isn't listed here, you have two options:

1. If the model is OpenAI compatible, use `ChatOpenAI()` with the appropriate `base_url` and `api_key` (see [`ChatGithub`](https://github.com/posit-dev/chatlas/blob/main/chatlas/_github.py) for a reference).
2. If you're motivated, implement a new provider by subclassing [`Provider`](https://github.com/posit-dev/chatlas/blob/main/chatlas/_provider.py) and implementing the required methods.


## Model choice

If you're using chatlas inside your organisation, you'll be limited to what your org allows, which is likely to be one provided by a big cloud provider (e.g. `ChatAzureOpenAI()` and `ChatBedrockAnthropic()`). If you're using chatlas for your own personal exploration, you have a lot more freedom so we have a few recommendations to help you get started:

- `ChatOpenAI()` or `ChatAnthropic()` are both good places to start. `ChatOpenAI()` defaults to **GPT-4o**, but you can use `model = "gpt-4o-mini"` for a cheaper lower-quality model, or `model = "o1-mini"` for more complex reasoning.  `ChatAnthropic()` is similarly good; it defaults to **Claude 3.5 Sonnet** which we have found to be particularly good at writing code.

- `ChatGoogle()` is great for large prompts, because it has a much larger context window than other models. It allows up to 1 million tokens, compared to Claude 3.5 Sonnet's 200k and GPT-4o's 128k.

- `ChatOllama()`, which uses [Ollama](https://ollama.com), allows you to run models on your own computer. The biggest models you can run locally aren't as good as the state of the art hosted models, but they also don't share your data and and are effectively free.

## Using chatlas

You can chat via `chatlas` in several different ways, depending on whether you are working interactively or programmatically. They all start with creating a new chat object:

```python
from chatlas import ChatOpenAI

chat = ChatOpenAI(
  model = "gpt-4o",
  system_prompt = "You are a friendly but terse assistant.",
)
```

### Interactive console

From a `chat` instance, it's simple to start a web-based or terminal-based chat console, which is great for testing the capabilities of the model. In either case, responses stream in real-time, and context is preserved across turns.

```python
chat.app()
```

<div align="center">
<img width="500" alt="A web app for chatting with an LLM via chatlas" src="https://github.com/user-attachments/assets/e43f60cb-3686-435a-bd11-8215cb024d2e" class="border rounded">
</div>


Or, if you prefer to work from the terminal:

```python
chat.console()
```

```
Entering chat console. Press Ctrl+C to quit.

?> Who created Python?

Python was created by Guido van Rossum. He began development in the late 1980s and released the first version in 1991. 

?> Where did he develop it?

Guido van Rossum developed Python while working at Centrum Wiskunde & Informatica (CWI) in the Netherlands.     
```


### The `.chat()` method

For a more programmatic approach, you can use the `.chat()` method to ask a question and get a response. By default, the response prints to a [rich](https://github.com/Textualize/rich) console as it streams in:

```python
chat.chat("What preceding languages most influenced Python?")
```

```
Python was primarily influenced by ABC, with additional inspiration from C,
Modula-3, and various other languages.
```

To ask a question about an image, pass one or more additional input arguments using `content_image_file()` and/or `content_image_url()`:

```python
from chatlas import content_image_url

chat.chat(
    content_image_url("https://www.python.org/static/img/python-logo.png"),
    "Can you explain this logo?"
)
```

```
The Python logo features two intertwined snakes in yellow and blue,
representing the Python programming language. The design symbolizes...
```

To get the full response as a string, use the built-in `str()` function. Optionally, you can also suppress the rich console output by setting `echo="none"`:

```python
response = chat.chat("Who is Posit?", echo="none")
print(str(response))
```

As we'll see in later articles, `echo="all"` can also be useful for debugging, as it shows additional information, such as tool calls.

### The `.stream()` method

If you want to do something with the response in real-time (i.e., as it arrives in chunks), use the `.stream()` method. This method returns an iterator that yields each chunk of the response as it arrives:

```python
response = chat.stream("Who is Posit?")
for chunk in response:
    print(chunk, end="")
```

The `.stream()` method can also be useful if you're [building a chatbot](https://posit-dev.github.io/chatlas/web-apps.html) or other programs that needs to display responses as they arrive.


### Tool calling 

Tool calling is as simple as passing a function with type hints and docstring to `.register_tool()`.

```python
import sys

def get_current_python_version() -> str:
    """Get the current version of Python."""
    return sys.version

chat.register_tool(get_current_python_version)
chat.chat("What's the current version of Python?")
```

```
The current version of Python is 3.13.
```

Learn more in the [tool calling article](https://posit-dev.github.io/chatlas/tool-calling.html)

### Structured data

Structured data (i.e., structured output) is as simple as passing a [pydantic](https://docs.pydantic.dev/latest/) model to `.extract_data()`.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

chat.extract_data(
    "My name is Susan and I'm 13 years old", 
    data_model=Person,
)
```

```
{'name': 'Susan', 'age': 13}
```

Learn more in the [structured data article](https://posit-dev.github.io/chatlas/structured-data.html)

### Export chat

Easily get a full markdown or HTML export of a conversation:

```python
chat.export("index.html", title="Python Q&A")
```

If the export doesn't have all the information you need, you can also access the full conversation history via the `.get_turns()` method:

```python
chat.get_turns()
```

And, if the conversation is too long, you can specify which turns to include:

```python
chat.export("index.html", turns=chat.get_turns()[-5:])
```

### Async

`chat` methods tend to be synchronous by default, but you can use the async flavor by appending `_async` to the method name:

```python
import asyncio

async def main():
    await chat.chat_async("What is the capital of France?")

asyncio.run(main())
```

### Typing support

`chatlas` has full typing support, meaning that, among other things, autocompletion just works in your favorite editor:

<div align="center">
<img width="500" alt="Autocompleting model options in ChatOpenAI" src="https://github.com/user-attachments/assets/163d6d8a-7d58-422d-b3af-cc9f2adee759" class="rounded">
</div>



### Troubleshooting

Sometimes things like token limits, tool errors, or other issues can cause problems that are hard to diagnose. 
In these cases, the `echo="all"` option is helpful for getting more information about what's going on under the hood.

```python
chat.chat("What is the capital of France?", echo="all")
```

This shows important information like tool call results, finish reasons, and more.

If the problem isn't self-evident, you can also reach into the `.get_last_turn()`, which contains the full response object, with full details about the completion.


<div align="center">
  <img width="500" alt="Turn completion details with typing support" src="https://github.com/user-attachments/assets/eaea338d-e44a-4e23-84a7-2e998d8af3ba" class="rounded">
</div>


For monitoring issues in a production (or otherwise non-interactive) environment, you may want to enabling logging. Also, since `chatlas` builds on top of packages like `anthropic` and `openai`, you can also enable their debug logging to get lower-level information, like HTTP requests and response codes.

```shell
$ export CHATLAS_LOG=info
$ export OPENAI_LOG=info
$ export ANTHROPIC_LOG=info
```

### Next steps

If you're new to world LLMs, you might want to read the [Get Started](https://posit-dev.github.io/chatlas/get-started.html) guide, which covers some basic concepts and terminology.

Once you're comfortable with the basics, you can explore more in-depth topics like [prompt design](https://posit-dev.github.io/chatlas/prompt-design.html) or the [API reference](https://posit-dev.github.io/chatlas/reference/index.html).
