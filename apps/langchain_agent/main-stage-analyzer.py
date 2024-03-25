import os
import sys
import typing
from dotenv import load_dotenv

from tools.contacts import get_all_contacts
from tools.vocode import call_phone_number
from tools.word_of_the_day import word_of_the_day
from vocode.turn_based.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.turn_based.synthesizer.gtts_synthesizer import GTTSSynthesizer
from langchain.memory import ConversationBufferMemory
from salesgpt.chains import StageAnalyzerChain, SalesConversationChain
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools

from langchain_community.chat_models import ChatLiteLLM


from stdout_filterer import RedactPhoneNumbers

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from salesgpt.agents import SalesGPT
from salesgpt.tools import get_tools

if __name__ == "__main__":
    # Redirect stdout to our custom class
    sys.stdout = typing.cast(typing.TextIO, RedactPhoneNumbers(sys.stdout))

    OBJECTIVE = (
        #input("Objective: ")
        #or 
        # "Find a random person in my contacts and tell them a joke"
        "Call Shai at +972545579687 and sell him a pen by asking him questions"
    )
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4")  # type: ignore
    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo")
    tools = get_tools("examples/sample_product_catalog.txt")
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
    
    sales_gpt = SalesGPT.from_llm_call_phone(llm=llm, 
                                             prompt=prompt,
                                             tools=[get_all_contacts, call_phone_number, word_of_the_day])
    while True:                                             
        sales_gpt.step()
    # sales_gpt.invoke(OBJECTIVE)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # # Logging of LLMChains
    # verbose = True
    # agent = initialize_agent(
    #     tools=[get_all_contacts, call_phone_number, word_of_the_day],
    #     llm=llm,
    #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #     verbose=verbose,
    #     memory=memory,
    # )
    # agent.run(OBJECTIVE)
