# Description

This is a small repo that lets you see how to configure llm internals at runtime in the context of a LangChain agent. It is setup to run with Azure OpenAI, and assumes you have a **gpt-4-turbo 1106-preview** model, and a **gpt-35-turbo 0613 or later** model deployed in Azure OpenAI.

Agent Features:
    - one tool in the demo, DuckDuckGo search
    - chat history *and* entity memory
    - llm is configurable at runtime, see main.py/run_agent ~ln 29:
    
        ```{python}
        agent.agent.runnable.steps[2].bound = agent.agent.runnable.steps[2].bound.with_config(configurable={"llm": llm_choice})
        ```


# Setup
1. Create a virtual environment, assumes you have python 3.10 or later
    -  `python -m venv .venv`
1. activate env
    - e.g. on windows using git bash: `source /.venv/Scripts/activate`
1. install requirements in venv
    - `pip install -r requirements.txt`
1. add variables to .env.sample and rename to .env
1. run demo
    - `python main.py`

