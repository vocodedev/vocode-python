from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from langchain.agents import (AgentExecutor, LLMSingleActionAgent,
                              create_openai_tools_agent)
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.agents import (_convert_agent_action_to_messages,
                                   _convert_agent_observation_to_messages)
from langchain_core.language_models.llms import create_base_retry_decorator
from litellm import acompletion
from pydantic import Field

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.custom_invoke import CustomAgentExecutor
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base


def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    """
    Creates a retry decorator for handling OpenAI API errors.

    This function creates a retry decorator that will retry a function call
    if it raises any of the specified OpenAI API errors. The maximum number of retries
    is determined by the 'max_retries' attribute of the 'llm' object.

    Args:
        llm (Any): An object that has a 'max_retries' attribute specifying the maximum number of retries.

    Returns:
        Callable[[Any], Any]: A retry decorator.
    """
    import openai

    errors = [
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[CustomAgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "gpt-3.5-turbo-0613"  # TODO - make this an env variable

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        """
        Retrieves the conversation stage based on the provided key.

        This function uses the key to look up the corresponding conversation stage in the conversation_stage_dict dictionary.
        If the key is not found in the dictionary, it defaults to "1".

        Args:
            key (str): The key to look up in the conversation_stage_dict dictionary.

        Returns:
            str: The conversation stage corresponding to the key, or "1" if the key is not found.
        """
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        """
        Property that returns a list of input keys.

        This property is currently set to return an empty list. It can be overridden in a subclass to return a list of keys
        that are used to extract input data from a dictionary.

        Returns:
            List[str]: An empty list.
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """
        Property that returns a list of output keys.

        This property is currently set to return an empty list. It can be overridden in a subclass to return a list of keys
        that are used to extract output data from a dictionary.

        Returns:
            List[str]: An empty list.
        """
        return []

    @time_logger
    def seed_agent(self):
        """
        This method seeds the conversation by setting the initial conversation stage and clearing the conversation history.

        The initial conversation stage is retrieved using the key "1". The conversation history is reset to an empty list.

        Returns:
            None
        """
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        """
        Determines the current conversation stage based on the conversation history.

        This method uses the stage_analyzer_chain to analyze the conversation history and determine the current stage.
        The conversation history is joined into a single string, with each entry separated by a newline character.
        The current conversation stage ID is also passed to the stage_analyzer_chain.

        The method then prints the determined conversation stage ID and retrieves the corresponding conversation stage
        from the conversation_stage_dict dictionary using the retrieve_conversation_stage method.

        Finally, the method prints the determined conversation stage.

        Returns:
            None
        """
        print(f"Conversation Stage ID before analysis: {self.conversation_stage_id}")
        print("Conversation history:")
        print(self.conversation_history)
        stage_analyzer_output = self.stage_analyzer_chain.invoke(
            input={
                "conversation_history": "\n".join(self.conversation_history).rstrip(
                    "\n"
                ),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join(
                    [
                        str(key) + ": " + str(value)
                        for key, value in CONVERSATION_STAGES.items()
                    ]
                ),
            },
            return_only_outputs=False,
        )
        print("Stage analyzer output")
        print(stage_analyzer_output)
        self.conversation_stage_id = stage_analyzer_output.get("text")

        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        """
        Processes the human input and appends it to the conversation history.

        This method takes the human input as a string, formats it by adding "User: " at the beginning and " <END_OF_TURN>" at the end, and then appends this formatted string to the conversation history.

        Args:
            human_input (str): The input string from the human user.

        Returns:
            None
        """
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(self, stream: bool = False):
        """
        Executes a step in the conversation. If the stream argument is set to True,
        it returns a streaming generator object for manipulating streaming chunks in downstream applications.
        If the stream argument is set to False, it calls the _call method with an empty dictionary as input.

        Args:
            stream (bool, optional): A flag indicating whether to return a streaming generator object.
            Defaults to False.

        Returns:
            Generator: A streaming generator object if stream is set to True. Otherwise, it returns None.
        """
        if not stream:
            return self._call(inputs={})
        else:
            return self._streaming_generator()

    @time_logger
    async def astep(self, stream: bool = False):
        """
        Executes an asynchronous step in the conversation.

        If the stream argument is set to False, it calls the _acall method with an empty dictionary as input.
        If the stream argument is set to True, it returns a streaming generator object for manipulating streaming chunks in downstream applications.

        Args:
            stream (bool, optional): A flag indicating whether to return a streaming generator object.
            Defaults to False.

        Returns:
            Generator: A streaming generator object if stream is set to True. Otherwise, it returns None.
        """
        if not stream:
            self._acall(inputs={})
        else:
            return await self._astreaming_generator()

    @time_logger
    def acall(self, *args, **kwargs):
        """
        This method is currently not implemented.

        Parameters
        ----------
        \*args : tuple
            Variable length argument list.
        \*\*kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            Indicates that this method has not been implemented yet.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    @time_logger
    def _prep_messages(self):
        """
        Prepares a list of messages for the streaming generator.

        This method prepares a list of messages based on the current state of the conversation.
        The messages are prepared using the 'prep_prompts' method of the 'sales_conversation_utterance_chain' object.
        The prepared messages include details about the current conversation stage, conversation history, salesperson's name and role,
        company's name, business, values, conversation purpose, and conversation type.

        Returns:
            list: A list of prepared messages to be passed to a streaming generator.
        """

        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            pass
            # print("\033[92m" + inception_messages[0].content + "\033[0m")
        return [message_dict]

    @time_logger
    def _streaming_generator(self):
        """
        Generates a streaming generator for partial LLM output manipulation.

        This method is used when the sales agent needs to take an action before the full LLM output is available.
        For example, when performing text to speech on the partial LLM output. The method returns a streaming generator
        which can manipulate partial output from an LLM in-flight of the generation.

        Returns
        -------
        generator
            A streaming generator for manipulating partial LLM output.

        Examples
        --------
        >>> streaming_generator = self._streaming_generator()
        >>> for chunk in streaming_generator:
        ...     print(chunk)
        Chunk 1, Chunk 2, ... etc.

        See Also
        --------
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        """
        Use tenacity to retry the async completion call.

        This method uses the tenacity library to retry the asynchronous completion call in case of failure.
        It creates a retry decorator using the '_create_retry_decorator' method and applies it to the
        '_completion_with_retry' function which makes the actual asynchronous completion call.

        Parameters
        ----------
        llm : Any
            The language model to be used for the completion.
        \*\*kwargs : Any
            Additional keyword arguments to be passed to the completion function.

        Returns
        -------
        Any
            The result of the completion function call.

        Raises
        ------
        Exception
            If the completion function call fails after the maximum number of retries.
        """
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self):
        """
        Asynchronous generator to reduce I/O blocking when dealing with multiple
        clients simultaneously.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation. This is useful in scenarios where the sales agent wants to take an action
        before the full LLM output is available. For instance, if we want to do text to speech on the partial LLM output.

        Returns
        -------
        AsyncGenerator
            A streaming generator which can manipulate partial output from an LLM in-flight of the generation.

        Examples
        --------
        >>> streaming_generator = self._astreaming_generator()
        >>> async for chunk in streaming_generator:
        >>>     await chunk ...
        Out: Chunk 1, Chunk 2, ... etc.

        See Also
        --------
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.sales_conversation_utterance_chain.llm,
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes one step of the sales agent.

        This function overrides the input temporarily with the current state of the conversation,
        generates the agent's utterance using either the sales agent executor or the sales conversation utterance chain,
        adds the agent's response to the conversation history, and returns the AI message.

        Parameters
        ----------
        inputs : Dict[str, Any]
            The initial inputs for the sales agent.

        Returns
        -------
        Dict[str, Any]
            The AI message generated by the sales agent.

        """
        # override inputs temporarily
        inputs = {
            "input": "",
            "conversation_stage": self.current_conversation_stage,
            "conversation_history": "\n".join(self.conversation_history),
            "salesperson_name": self.salesperson_name,
            "salesperson_role": self.salesperson_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
        }

        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.invoke(inputs)
            output = ai_message["output"]
        else:
            ai_message = self.sales_conversation_utterance_chain.invoke(
                inputs, return_intermediate_steps=True
            )
            output = ai_message["text"]

        # Add agent's response to conversation history
        agent_name = self.salesperson_name
        output = agent_name + ": " + output
        if "<END_OF_TURN>" not in output:
            output += " <END_OF_TURN>"
        self.conversation_history.append(output)

        if self.verbose:
            tool_status = "USE TOOLS INVOKE:" if self.use_tools else "WITHOUT TOOLS:"
            print(f"{tool_status}\n#\n#\n#\n#\n------------------")
            print(f"AI Message: {ai_message}")
            print()
            print(f"Output: {output.replace('<END_OF_TURN>', '')}")

        return ai_message

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """
        Class method to initialize the SalesGPT Controller from a given ChatLiteLLM instance.

        This method sets up the stage analyzer chain and sales conversation utterance chain. It also checks if custom prompts
        are to be used and if tools are to be set up for the agent. If tools are to be used, it sets up the knowledge base,
        gets the tools, sets up the prompt, and initializes the agent with the tools. If tools are not to be used, it sets
        the sales agent executor and knowledge base to None.

        Parameters
        ----------
        llm : ChatLiteLLM
            The ChatLiteLLM instance to initialize the SalesGPT Controller from.
        verbose : bool, optional
            If True, verbose output is enabled. Default is False.
        \*\*kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SalesGPT
            The initialized SalesGPT Controller.
        """
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        # Handle custom prompts
        use_custom_prompt = kwargs.pop("use_custom_prompt", False)
        custom_prompt = kwargs.pop("custom_prompt", None)

        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm,
            verbose=verbose,
            use_custom_prompt=use_custom_prompt,
            custom_prompt=custom_prompt,
        )

        # Handle tools
        use_tools_value = kwargs.pop("use_tools", False)
        if isinstance(use_tools_value, str):
            if use_tools_value.lower() not in ["true", "false"]:
                raise ValueError("use_tools must be 'True', 'False', True, or False")
            use_tools = use_tools_value.lower() == "true"
        elif isinstance(use_tools_value, bool):
            use_tools = use_tools_value
        else:
            raise ValueError(
                "use_tools must be a boolean or a string ('True' or 'False')"
            )
        sales_agent_executor = None
        knowledge_base = None

        if use_tools:
            product_catalog = kwargs.pop("product_catalog", None)
            tools = get_tools(product_catalog)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
            tool_names = [tool.name for tool in tools]
            output_parser = SalesConvoOutputParser(
                ai_prefix=kwargs.get("salesperson_name", ""), verbose=verbose
            )
            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            sales_agent_executor = CustomAgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools,
                tools=tools,
                verbose=verbose,
                return_intermediate_steps=True,
            )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            model_name=llm.model,
            verbose=verbose,
            use_tools=use_tools,
            **kwargs,
        )


    @classmethod
    @time_logger
    def from_llm_call_phone(cls, llm: ChatLiteLLM, prompt, tools) -> "SalesGPT":
        """
        Class method to initialize the SalesGPT Controller from a given ChatLiteLLM instance.

        This method sets up the stage analyzer chain and sales conversation utterance chain. It also checks if custom prompts
        are to be used and if tools are to be set up for the agent. If tools are to be used, it sets up the knowledge base,
        gets the tools, sets up the prompt, and initializes the agent with the tools. If tools are not to be used, it sets
        the sales agent executor and knowledge base to None.

        Parameters
        ----------
        llm : ChatLiteLLM
            The ChatLiteLLM instance to initialize the SalesGPT Controller from.
        verbose : bool, optional
            If True, verbose output is enabled. Default is False.
        \*\*kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SalesGPT
            The initialized SalesGPT Controller.
        """
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=True)
        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=True,
            use_custom_prompt=False,
            custom_prompt=None,
        )

        verbose = True
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
        tool_names = [tool.name for tool in tools]
        output_parser = SalesConvoOutputParser(
            ai_prefix="", verbose=verbose
        )
        sales_agent_with_tools = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        sales_agent_executor = CustomAgentExecutor.from_agent_and_tools(
            agent=sales_agent_with_tools,
            tools=tools,
            verbose=verbose,
            return_intermediate_steps=True,
        )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=None,
            model_name=llm.model,
            verbose=verbose,
            use_tools=True,
        )
