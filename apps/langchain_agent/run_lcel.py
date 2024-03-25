import os
import json
from dotenv import load_dotenv 

from langchain.agents import AgentExecutor, initialize_agent
from salesgpt.tools import initialize_tool

from langchain_community.chat_models import ChatLiteLLM

from salesgpt.agents import SalesGPT
# Import your tools and agent classes here

def load_lcel_config(config_path):
    """Load LCEL configuration from a file."""
    with open(config_path, 'r') as file:
        return json.load(file)

def initialize_from_lcel_config(lcel_config):
    """Initialize and return an agent based on an LCEL configuration."""
    # Load .env variables
    load_dotenv()

    # Setup environment variables if specified in LCEL
    for key, value in lcel_config.get('environment', {}).items():
        os.environ[key] = value
    
    # Initialize LLM
    llm_config = lcel_config['agent']['llm']
    # llm = LLMFactory.create(llm_config['type'], **llm_config['params'])
    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo")

    # Initialize tools (assuming a function to initialize tools based on their type exists)
    tools = [initialize_tool(tool_conf) for tool_conf in lcel_config['agent'].get('tools', [])]

    # Extract additional agent configuration
    agent_config = lcel_config['agent'].get('sales_agent_config', {})

    # Initialize SalesGPT agent (this part may need to be adjusted based on your setup)
    agent = SalesGPT.from_llm(llm=llm, **agent_config)

    return agent

def main(config_path):
    """Main function to load the LCEL config and run the agent."""
    lcel_config = load_lcel_config(config_path)
    agent = initialize_from_lcel_config(lcel_config)
    
    # Assuming the SalesGPT agent has a method 'run' to start its process
    agent.run()

if __name__ == "__main__":
    config_path = './lcel-config.json'  # Update with the actual path to your LCEL config file
    main(config_path)
