# chatlas

Easily chat with various LLM models from Ollama, Anthropic, OpenAI, and more.
`chatlas` is intentionally minimal -- making it easy to get started, while also supporting important features like streaming, tool calling, images, async, and more.

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```bash
pip install git+https://github.com/cpsievert/chatlas
```

After installing, you'll want to pick a [model provider](#model-providers), and get [credentials](#managing-credentials) set up (if necessary). Here, we demonstrate usage with OpenAI, but the concepts here apply to other implementations as well.

## Model providers

`chatlas` supports a variety of model providers:

* Anthropic (Claude): [`ChatAnthropic()`](https://cpsievert.github.io/chatlas/reference/ChatAnthropic.html).
* GitHub model marketplace: [`ChatGithub()`](https://cpsievert.github.io/chatlas/reference/ChatGithub.html).
* Google (Gemini): [`ChatGoogle()`](https://cpsievert.github.io/chatlas/reference/ChatGoogle.html).
* Groq: [`ChatGroq()`](https://cpsievert.github.io/chatlas/reference/ChatGroq.html).
* Ollama local models: [`ChatOllama()`](https://cpsievert.github.io/chatlas/reference/ChatOllama.html).
* OpenAI: [`ChatOpenAI()`](https://cpsievert.github.io/chatlas/reference/ChatOpenAI.html).
* perplexity.ai: [`ChatPerplexity()`](https://cpsievert.github.io/chatlas/reference/ChatPerplexity.html).

As well as enterprise cloud providers:

* AWS Bedrock: [`ChatBedrockAnthropic()`](https://cpsievert.github.io/chatlas/reference/ChatBedrockAnthropic.html).
* Azure OpenAI: [`ChatAzureOpenAI()`](https://cpsievert.github.io/chatlas/reference/ChatAzureOpenAI.html).


## Model choice

If you're using chatlas inside your organisation, you'll typically need to use whatever you're allowed to. If you're using chatlas for your own personal exploration, we recommend starting with:

`ChatOpenAI()`, which currently defaults to `model="gpt-4o-mini"`. You might want to try `model="gpt-4o"` for more demanding task and if you want to force complex reasoning, `model="o1-mini"`.

`ChatAnthropic()`, which defaults to Claude 3.5 Sonnet. This currently appears to be the best model for code generation.

If you want to put a lot of data in the prompt, try `ChatGoogle()` which defaults to Gemini 1.5 Flash and supports 1 million tokens, compared to 200k for Claude 3.5 Sonnet and 128k for GPT 4o mini.

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

From a `chat` instance, you can start an interacitve, multi-turn, conversation in the console (via `.console()`) or in a browser (via `.app()`).

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

The chat console is useful for quickly exploring the capabilities of the model, especially when you've customized the chat object with tool integrations (covered later).

The chat app is similar to the chat console, but it runs in your browser. It's useful if you need more interactive capabilities like easy copy-paste.

```python
chat.app()
```

<div align="center">
<img width="667" alt="Screenshot 2024-11-06 at 11 43 34 AM" src="https://github.com/user-attachments/assets/e43f60cb-3686-435a-bd11-8215cb024d2e">
</div>


Again, keep in mind that the chat object retains state, so when you enter the chat console, any previous interactions with that chat object are still part of the conversation, and any interactions you have in the chat console will persist even after you exit back to the Python prompt.


### The `.chat()` method

For a more programmatic approach, you can use the `.chat()` method to ask a question and get a response. If you're in a REPL (e.g., Jupyter, IPython, etc), the result of `.chat()` is automatically displayed using a [rich](https://github.com/Textualize/rich) console.

```python
chat.chat("What preceding languages most influenced Python?")
```

```
Python was primarily influenced by ABC, with additional inspiration from C,
Modula-3, and various other languages.
```

If you're not in a REPL (e.g., a non-interactive Python script), you can explicitly `.display()` the response:

```python
response = chat.chat("What is the Python programming language?")
response.display()
```

The `response` is also an iterable, so you can loop over it to get the response in streaming chunks:

```python
result = ""
for chunk in response:
    result += chunk
```

Or, if you just want the full response as a string, use the built-in `str()` function:

```python
str(response)
```


### Vision (Image Input)

To ask questions about images, you can pass one or more additional input arguments using `content_image_file()` and/or `content_image_url()`:

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

The `content_image_url()` function takes a URL to an image file and sends that URL directly to the API. The `content_image_file()` function takes a path to a local image file and encodes it as a base64 string to send to the API. Note that by default, `content_image_file()` automatically resizes the image to fit within 512x512 pixels; set the `resize` parameter to "high" if higher resolution is needed.


### Conversation history

Remember that regardless of how we interact with the model, the `chat` instance retains the conversation history, which you can access at any time:

```python
chat.turns()
```

Each turn represents a either a user's input or a model's response. It holds all the avaliable information about content and metadata of the turn. This can be useful for debugging, logging, or for building more complex conversational interfaces.

### Learn more

For more information on the available methods and parameters, check out the [API reference](https://cpsievert.github.io/chatlas/reference/index.html).
