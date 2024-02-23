import time

from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool
from langchain_community.callbacks import get_openai_callback, OpenAICallbackHandler
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

from src.llm import model
from src.agent import create_agent

def run_agent(llm_choice: str, agent: AgentExecutor, user_message, **kwargs):
    """
    Runner function for the agent

    Args:
        llm_choice (str): the choice of language model to use
        agent (AgentExecutor): the agent to be used for generative content
        task (str): the task to be performed
        prompt_name (str): the name of the prompt to be used
        config (Dict[str, Any]): the configuration dictionary, typically from a yaml file
        **kwargs: additional arguments to be passed to the fill_placeholders function

    Returns:
        Tuple[Any]: a tuple containing the result, callback, and duration
    """
    with get_openai_callback() as cb, Timer() as tmr:
        # invoke with gpt3x for faster response, gpt4 for more robust reasoning response
        agent.agent.runnable.steps[2].bound = agent.agent.runnable.steps[2].bound.with_config(configurable={"llm": llm_choice})

        # invoke the agent
        result = agent.invoke({"input": user_message})

    return result, cb, tmr.interval

class Timer:
    """
    Timer class to time execution of code blocks.
    
    Usage:
        with Timer() as t:
            # code block to time
        print(f"Time elapsed: {t.interval}")
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


if __name__ == "__main__":
    # simple ddg search tool
    # duck duck go generalist search
    wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", time="6m", max_results=10)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="web")
    tools = [
        Tool(
            name="DuckDuckGoSearch",
            description="Search DuckDuckGo. Useful for when you need to acquire recent news. Ask targeted questions",
            tool_type="search",
            func=search.run
        )
    ]

    agent = create_agent(llm = model, tools = tools, extra_prompt_messages = []
                , verbose = True, handle_parsing_errors = True, callbacks = [OpenAICallbackHandler()])
    
    # example use with gpt4
    prompt = "Summarize the personal of Severus Snape from the Harry Potter series in 100 words or less."
    print(f"Prompt: {prompt}")
    print(f'running agent with gpt4...')
    result, cb, duration = run_agent('gpt4', agent, prompt)
    print(f" OpenAI Callback Results: {cb}, Run Duration (seconds): {duration}")
    print(result['output'])
    print('*'*50)

    # example use with gpt3x
    # clear out memory and entities first
    agent.memory.chat_memory.clear()
    agent.memory.entity_store.clear()


    print(f'running agent with gpt35turbo...')
    result, cb, duration = run_agent('gpt35turbo', agent, prompt)
    print(f" OpenAI Callback Results: {cb}, Run Duration (seconds): {duration}")
    print(result['output'])
    print('*'*50)