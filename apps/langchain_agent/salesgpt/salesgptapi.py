import asyncio
import json
import re

from langchain_community.chat_models import ChatLiteLLM

from salesgpt.agents import SalesGPT


class SalesGPTAPI:
    def __init__(
        self,
        config_path: str,
        verbose: bool = True,
        max_num_turns: int = 20,
        model_name: str = "gpt-3.5-turbo",
        product_catalog: str = "examples/sample_product_catalog.txt",
        use_tools=True,
    ):
        self.config_path = config_path
        self.verbose = verbose
        self.max_num_turns = max_num_turns
        self.llm = ChatLiteLLM(temperature=0.2, model_name=model_name)
        self.product_catalog = product_catalog
        self.conversation_history = []
        self.use_tools = use_tools
        self.sales_agent = self.initialize_agent()
        self.current_turn = 0

    def initialize_agent(self):
        config = {"verbose": self.verbose}
        if self.config_path:
            with open(self.config_path, "r") as f:
                config.update(json.load(f))
            if self.verbose:
                print(f"Loaded agent config: {config}")
        else:
            print("Default agent config in use")

        if self.use_tools:
            print("USING TOOLS")
            config.update(
                {
                    "use_tools": True,
                    "product_catalog": self.product_catalog,
                    "salesperson_name": "Ted Lasso"
                    if not self.config_path
                    else config.get("salesperson_name", "Ted Lasso"),
                }
            )

        sales_agent = SalesGPT.from_llm(self.llm, **config)

        print(f"SalesGPT use_tools: {sales_agent.use_tools}")
        sales_agent.seed_agent()
        return sales_agent

    def do(self, human_input=None):
        self.current_turn += 1
        current_turns = self.current_turn
        if current_turns >= self.max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            return [
                "BOT",
                "In case you'll have any questions - just text me one more time!",
            ]

        if human_input is not None:
            self.sales_agent.human_step(human_input)

        ai_log = self.sales_agent.step(stream=False)
        self.sales_agent.determine_conversation_stage()
        # TODO - handle end of conversation in the API - send a special token to the client?
        if self.verbose:
            print("=" * 10)
            print(ai_log)
        if (
            self.sales_agent.conversation_history
            and "<END_OF_CALL>" in self.sales_agent.conversation_history[-1]
        ):
            print("Sales Agent determined it is time to end the conversation.")
            # strip end of call for now
            self.sales_agent.conversation_history[
                -1
            ] = self.sales_agent.conversation_history[-1].replace("<END_OF_CALL>", "")

        reply = (
            self.sales_agent.conversation_history[-1]
            if self.sales_agent.conversation_history
            else ""
        )

        if (
            self.use_tools
            and ai_log["intermediate_steps"][1]["outputs"]["intermediate_steps"]
            is not []
        ):
            try:
                res_str = ai_log["intermediate_steps"][1]["outputs"][
                    "intermediate_steps"
                ][0]
                tool_search_result = res_str[0]
                agent_action = res_str[0]
                tool, tool_input, log = (
                    agent_action.tool,
                    agent_action.tool_input,
                    agent_action.log,
                )
                actions = re.search(r"Action: (.*?)[\n]*Action Input: (.*)", log)
                action_input = actions.group(2)
                action_output = ai_log["intermediate_steps"][1]["outputs"][
                    "intermediate_steps"
                ][0][1]
            except:
                tool, tool_input, action, action_input, action_output = (
                    "",
                    "",
                    "",
                    "",
                    "",
                )
        else:
            tool, tool_input, action, action_input, action_output = "", "", "", "", ""

        print(reply)
        payload = {
            "bot_name": reply.split(": ")[0],
            "response": ": ".join(reply.split(": ")[1:]).rstrip("<END_OF_TURN>"),
            "conversational_stage": self.sales_agent.current_conversation_stage,
            "tool": tool,
            "tool_input": tool_input,
            "action_output": action_output,
            "action_input": action_input,
        }
        return payload

    async def do_stream(self, conversation_history: [str], human_input=None):
        # TODO
        current_turns = len(conversation_history) + 1
        if current_turns >= self.max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            yield [
                "BOT",
                "In case you'll have any questions - just text me one more time!",
            ]
            raise StopAsyncIteration

        self.sales_agent.seed_agent()
        self.sales_agent.conversation_history = conversation_history

        if human_input is not None:
            self.sales_agent.human_step(human_input)

        stream_gen = self.sales_agent.astep(stream=True)
        for model_response in stream_gen:
            for choice in model_response.choices:
                message = choice["delta"]["content"]
                if message is not None:
                    if "<END_OF_CALL>" in message:
                        print(
                            "Sales Agent determined it is time to end the conversation."
                        )
                        yield [
                            "BOT",
                            "In case you'll have any questions - just text me one more time!",
                        ]
                    yield message
                else:
                    continue
