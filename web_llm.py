from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(model_name="gpt-4",temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.

tools = load_tools(["serpapi"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
#agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
#agent.run("Wie hoch soll die CO2-Instensität des Portfolios der DZ BANK AG bis 2025 gesenkt werden?")
#agent.run("Wie hoch ist der Frauenanteil in Führungsgremien der DZ BANK AG auf allen Ebenen basierend auf dem Nachhaltigkeitsbericht 2022?")
agent.run("Wie hoch war der Personalaufwand des Unternehmens Dürr in 2022?")
