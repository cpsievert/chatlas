# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [UNRELEASED]

### New features

* Adds vLLM support via a new `ChatVLLM` class. (#24)

### Bug fixes

* `ChatOllama` no longer fails when a `OPENAI_API_KEY` environment variable is not set.
* `ChatOpenAI` now correctly includes the relevant `detail` on `ContentImageRemote()` input.


## [0.2.0] - 2024-12-11

First stable release of `chatlas`, see the website to learn more <https://posit-dev.github.io/chatlas/> 
