#!/usr/bin/python3

from os import environ
from huggingface_hub import login
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import ChatOpenAI

def CodeLlama(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 16384,
        "do_sample": False,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      max_length = 16384,
      do_sample = False,
      temperature = 0.8,
      top_p = 0.8,
      use_cache = True
    )
  return tokenizer, llm

def Qwen2(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'Qwen/Qwen2-7B-Instruct',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 131072,
        "do_sample": False,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = 'Qwen/Qwen2-7B-Instruct',
      task = 'text-generation',
      do_sample = False,
      temperature = 0.8,
      top_p = 0.8,
    )
  return tokenizer, llm