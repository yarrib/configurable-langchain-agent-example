import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.runnables import ConfigurableField
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
OPENAI_API_VERSION = os.environ.get('OPENAI_API_VERSION')


gpt35turbo = AzureChatOpenAI(temperature=0
                , deployment_name = 'gpt35turbo'
                , model = 'gpt-35-turbo'
                , openai_api_key = AZURE_OPENAI_API_KEY
                , azure_endpoint = AZURE_OPENAI_ENDPOINT
                , openai_api_version = OPENAI_API_VERSION
                , max_tokens = 400
                )


model = AzureChatOpenAI(temperature=0, deployment_name = 'gpt4'
                , model = 'gpt-4'
                # , model_version= '1106'
                , openai_api_key = AZURE_OPENAI_API_KEY
                , azure_endpoint = AZURE_OPENAI_ENDPOINT
                , openai_api_version = OPENAI_API_VERSION
                , max_tokens = 400
                ).configurable_alternatives(
    # give this an id of "llm"
    ConfigurableField(id="llm"),
    # default to gpt4
    default_key="gpt4",
    # give an alternative of gpt35turbo
    gpt35turbo=gpt35turbo
)