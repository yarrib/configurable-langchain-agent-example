from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool, tool

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.memory import ConversationEntityMemory

from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)

from langchain_community.callbacks import OpenAICallbackHandler
from langchain.callbacks.base import BaseCallbackManager

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage
from langchain_core.messages.tool import ToolMessage

from langchain_openai import AzureChatOpenAI

from typing import List, Union, Dict, Any

from .llm import gpt35turbo

system_message_template = """
You are an assistant to a human, powered by a large language model trained by OpenAI. \
    
You are designed to be able to assist with a wide range of tasks, \
from answering simple questions to providing in-depth explanations and \
discussions on a wide range of topics. As a language model, you are able \
to generate human-like text based on the input you receive, allowing you \
to engage in natural-sounding conversations and provide responses that are \
coherent and relevant to the topic at hand. \

You are constantly learning and improving, and your capabilities are constantly evolving. \
You are able to process and understand large amounts of text, and can use this knowledge to \
provide accurate and informative responses to a wide range of questions. You have access to some \
personalized information provided by the human in the Context section below. \
Additionally, you are able to generate your own text based on the input you receive, \
allowing you to engage in discussions and provide explanations and descriptions on a \
wide range of topics.

Overall, you are a powerful tool that can help with a wide \
range of tasks and provide valuable insights and information on a wide range of topics. \
Whether the human needs help with a specific question or just wants to have a conversation \
about a particular topic, you are here to assist. \

Context:
{entities}

"""
prompt = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'entities']
    , input_types={
        'chat_history': List[Union[AIMessage,HumanMessage,ChatMessage,SystemMessage,FunctionMessage,ToolMessage]]
        , 'agent_scratchpad': List[Union[AIMessage,HumanMessage,ChatMessage,SystemMessage,FunctionMessage,ToolMessage]]                                   
        , 'entities': Dict[str, Any]
                }
        , messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['entities']
                                                                      , template=system_message_template))
        , MessagesPlaceholder(variable_name='chat_history', optional=True)
        , HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))
        , MessagesPlaceholder(variable_name='agent_scratchpad')])

        

def create_agent(llm: AzureChatOpenAI
                 , tools = []
                 , prompt: ChatPromptTemplate = prompt
                 , extra_prompt_messages: List[Dict[str,str]] = [{}]
                 , verbose=True
                 , handle_parsing_errors=True
                 , callbacks: List[Any] = None) -> AgentExecutor:
    """
    Create an agent executor with the given language model and tools.
    
    Args:
        llm: The language model to use for the agent. defaults to gpt-35-turbo
        tools: A list of tools to use for the agent. defults to no tools, []
        prompt: The prompt (system message) to use for the agent. defaults to the default prompt from the module
        extra_prompt_messages: A list of messages to add to the memory before the agent starts. defaults to an empty list
        verbose: A boolean flag to indicate if the agent should be verbose. defaults to True
        handle_parsing_errors: A boolean flag to indicate if the agent should handle parsing errors. defaults to True

    Returns:
        An instance of AgentExecutor with the given language model and tools.

    """

    # setup our inital memory state
    memory = ConversationEntityMemory(llm=gpt35turbo, chat_history_key='chat_history', return_messages=True)
    try:
        for idx,d in enumerate(extra_prompt_messages):
            if idx % 2 == 0:
                _input = {"input": d['user']}
            else:
                _output = {"output": d['ai']}
                memory.load_memory_variables(_input)
                memory.save_context(_input, _output)
    except:
        pass

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor.from_agent_and_tools(agent=agent
                                    , tools=tools
                                    , memory=memory
                                    , verbose=verbose
                                    , handle_parsing_errors=handle_parsing_errors
                                    , callbacks=callbacks
                                    )