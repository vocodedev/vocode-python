import openai
from typing import List, Type
from vocode import getenv
from pydantic import BaseModel, Field
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionInput, ActionOutput, ActionType

DEFAULT_EMBEDDINGS_MODEL = "text-embedding-ada-002"


class PineconeLookupParameters(BaseModel):
    pinecone_index: str = Field(..., description="The name of the Pinecone index.")
    text: str = Field(..., description="The search text.")
    num_results: int = Field(..., description="The number of results to return.")


class PineconeLookupResponse(BaseModel):
    results: List


class PineconeLookup(BaseAction[PineconeLookupParameters, PineconeLookupResponse]):
    description: str = "Embeddings search using the Pinecone API."
    action_type: str = ActionType.PINECONE_LOOKUP.value
    parameters_type: Type[PineconeLookupParameters] = PineconeLookupParameters
    response_type: Type[PineconeLookupResponse] = PineconeLookupResponse

    async def run(
        self, action_input: ActionInput[PineconeLookupParameters]
    ) -> ActionOutput[PineconeLookupResponse]:
        import pinecone

        pinecone.init(
            api_key=getenv("PINECONE_API_KEY"),
            environment=getenv("PINECONE_ENVIRONMENT"),
        )

        index = pinecone.Index(action_input.params.pinecone_index)

        embed = openai.Embedding.create(
            input=[action_input.params.text], engine=DEFAULT_EMBEDDINGS_MODEL
        )["data"][0]["embedding"]
        res = index.query(
            [embed], top_k=action_input.params.num_results, include_metadata=True
        )

        return ActionOutput(
            action_type=action_input.action_type,
            response=PineconeLookupResponse(results=[x.to_dict() for x in res["matches"]]),
        )
