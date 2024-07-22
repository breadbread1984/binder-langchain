# Introduction

this project is a reimplement of binder with langchain and open sourced LLMs

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## run evaluation

it is hard to introduce huggingface's beam search into langchain framework, therefore the consistency of the LLM generation is not included in this implement

```shell
python3 main.py --dataset (tab_fact|mmqa|wikiq) --model (llama3|codellama|qwen2|codeqwen) [--locally]
```

