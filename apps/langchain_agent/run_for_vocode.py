import argparse
import json
import logging
import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM

from salesgpt.agents import SalesGPT

load_dotenv()  # loads .env file

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

# LangSmith settings section, set TRACING_V2 to "true" to enable it
# or leave it as it is, if you don't need tracing (more info in README)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = ""  # insert you project name here

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add arguments
    parser.add_argument(
        "--config", type=str, help="Path to agent config file", default=""
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbosity", default=False
    )
    parser.add_argument(
        "--max_num_turns",
        type=int,
        help="Maximum number of turns in the sales conversation",
        default=10,
    )

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns

    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo")

    if config_path == "":
        print("No agent config specified, using a standard config")
        # keep boolean as string to be consistent with JSON configs.
        USE_TOOLS = True
        sales_agent_kwargs = {
            "verbose": verbose,
            "use_tools": USE_TOOLS,
        }

        if USE_TOOLS:
            sales_agent_kwargs.update(
                {
                    "product_catalog": "examples/sample_product_catalog.txt",
                    "salesperson_name": "Ted Lasso",
                }
            )

        sales_agent = SalesGPT.from_llm(llm, **sales_agent_kwargs)
    else:
        try:
            with open(config_path, "r", encoding="UTF-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the config file {config_path}.")
            exit(1)

        print(f"Agent config {config}")
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

    sales_agent.seed_agent()
    sales_agent.run()
    
    
    
    # print("=" * 10)
    # cnt = 0
    # while cnt != max_num_turns:
    #     cnt += 1
    #     if cnt == max_num_turns:
    #         print("Maximum number of turns reached - ending the conversation.")
    #         break
    #     sales_agent.step()

    #     # end conversation
    #     if "<END_OF_CALL>" in sales_agent.conversation_history[-1]:
    #         print("Sales Agent determined it is time to end the conversation.")
    #         break
    #     human_input = input("Your response: ")
    #     sales_agent.human_step(human_input)
    #     print("=" * 10)
