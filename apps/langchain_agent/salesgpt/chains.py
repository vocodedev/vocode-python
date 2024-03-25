from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatLiteLLM

from salesgpt.logger import time_logger
from salesgpt.prompts import (SALES_AGENT_INCEPTION_PROMPT,
                              STAGE_ANALYZER_INCEPTION_PROMPT)


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = STAGE_ANALYZER_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=[
                "conversation_history",
                "conversation_stage_id",
                "conversation_stages",
            ],
        )
        print(f"STAGE ANALYZER PROMPT {prompt}")
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class SalesConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    @time_logger
    def from_llm(
        cls,
        llm: ChatLiteLLM,
        verbose: bool = True,
        use_custom_prompt: bool = False,
        custom_prompt: str = "You are an AI Sales agent, Call Shai at +972545579687 and sell him this pen by asking him question",
    ) -> LLMChain:
        """Get the response parser."""
        if use_custom_prompt:
            sales_agent_inception_prompt = custom_prompt
            prompt = PromptTemplate(
                template=sales_agent_inception_prompt,
                input_variables=[
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
        else:
            sales_agent_inception_prompt = SALES_AGENT_INCEPTION_PROMPT
            prompt = PromptTemplate(
                template=sales_agent_inception_prompt,
                input_variables=[
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
        return cls(prompt=prompt, llm=llm, verbose=verbose)
