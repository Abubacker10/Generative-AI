from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from crewai import Agent, Task, Crew, Process
from crewai_tools import CSVSearchTool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
from crewai import LLM
from crewai_tools import CodeInterpreterTool
import os
from dotenv import load_dotenv

load_dotenv()
# groq_api_key = "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"

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
os.environ['OPENAI_API_KEY'] = 'yeyeyey111'
llm = LLM(model="ollama/llama3.1:8b",
          base_url="http://localhost:11434")

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1")

# python_repl = PythonREPL()
# repl_tool = Tool(
#     name="python_repl",
#     description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#     func=python_repl.run,
# )

# tool = CSVSearchTool(csv=r"C:\Users\navab\Downloads\retail_sales_dataset.csv",
tool = CSVSearchTool(
    csv=r"C:\Users\navab\Downloads\retail_sales_dataset.csv",
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3.1",
                temperature=0.5,
                top_p=1
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="mixedbread-ai/mxbai-embed-large-v1"
                # title="Embeddings",
            ),
        ),
    )
)

data_analyst = Agent(
  role='data_analyst',
  goal='''Help user to get insights from the Data''',
  backstory='You are an expert data analyst with decades of experience.',
  allow_delegation=True,verbose=True,
  memory=True,
  llm='ollama/llama3.1',
  tools = [tool])

eda_agent=Agent(
    role='Code interpreter',
    backstory="This agent specializes in code interpretation and debugging.",
    goal='Complie and run the code and remove irrelavant characters such as ` backclips ',
    verbose=True,
    memory=True,
    llm='ollama/llama3.1',
    tools=[CodeInterpreterTool()],
    allow_delegation=False

)

data_analyst_task = Task(
  description=(
    "analyse the context of data"
    "understand the distribution and importance of features"
    "generate insights"
  ),
  expected_output='Generate a cleaned version of python for exploratory data analysis',
  tools=[tool],
  agent=data_analyst,
)

eda_task = Task(
  description=(
    "get the cleaned python code for eda"
  ),
  expected_output='run the code and get the results',
  tools=[CodeInterpreterTool()],
  agent=eda_agent,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

crew = Crew(
  agents=[data_analyst,eda_agent],
  tasks=[data_analyst_task, eda_task],
  process=Process.sequential,  
  llm=llm,# Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

result=crew.kickoff()
print(result)