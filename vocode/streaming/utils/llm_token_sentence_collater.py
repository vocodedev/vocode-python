from vocode.streaming.agent.agent_response import AgentResponse
from vocode.streaming.utils.worker import AbstractWorker, InterruptibleAgentResponseEvent


class LLMTokenSentenceCollater(AbstractWorker[InterruptibleAgentResponseEvent[AgentResponse]]):
    consumer: AbstractWorker[InterruptibleAgentResponseEvent[AgentResponse]]

    def __init__(self):
        self.buffer = ""
        self.function_name_buffer = ""
        self.function_args_buffer = ""
        self.is_post_period = False
        self.tokens_since_period = 0
        self.is_first = True

    def consume_nonblocking(self, item: InterruptibleAgentResponseEvent[AgentResponse]):
        if self.is_first:
            if sentry_span:
                sentry_span.finish()
            self.is_first = False
        if not token:
            continue
        if isinstance(token, str):
            buffer += token
            if len(buffer.strip().split()) < SHORT_SENTENCE_CUTOFF:
                continue
            if re.search(SENTENCE_ENDINGS_EXCEPT_PERIOD_PATTERN, token):
                # split on last occurrence of sentence ending
                matches = [
                    match for match in re.finditer(SENTENCE_ENDINGS_EXCEPT_PERIOD_PATTERN, buffer)
                ]
                last_match = matches[-1]
                split_point = last_match.start() + 1
                to_keep, to_return = buffer[split_point:], buffer[:split_point]
                if to_return.strip():
                    yield to_return.strip()
                buffer = to_keep
            elif "." in token:
                is_post_period = True
                tokens_since_period = 0

            if is_post_period and tokens_since_period > TOKENS_TO_GENERATE_PAST_PERIOD:
                sentences = split_sentences(buffer)
                if len(sentences) > 1:
                    yield " ".join(sentences[:-1])
                    buffer = sentences[-1]
                is_post_period = False
                tokens_since_period = 0
            else:
                tokens_since_period += 1

        elif isinstance(token, FunctionFragment):
            function_name_buffer += token.name
            function_args_buffer += token.arguments

    async def collate_response_async(
        conversation_id: str,
        gen: AsyncIterable[Union[str, FunctionFragment]],
        get_functions: Literal[True, False] = False,
        sentry_span: Optional[Span] = None,
    ) -> AsyncGenerator[
        Union[str, FunctionCall],
        None,
    ]:  # tuple of message to send and whether it's the final message
        buffer = ""
        function_name_buffer = ""
        function_args_buffer = ""
        is_post_period = False
        tokens_since_period = 0
        is_first = True
        async for token in gen:
            if is_first:
                if sentry_span:
                    sentry_span.finish()
                is_first = False
            if not token:
                continue
            if isinstance(token, str):
                buffer += token
                if len(buffer.strip().split()) < SHORT_SENTENCE_CUTOFF:
                    continue
                if re.search(SENTENCE_ENDINGS_EXCEPT_PERIOD_PATTERN, token):
                    # split on last occurrence of sentence ending
                    matches = [
                        match
                        for match in re.finditer(SENTENCE_ENDINGS_EXCEPT_PERIOD_PATTERN, buffer)
                    ]
                    last_match = matches[-1]
                    split_point = last_match.start() + 1
                    to_keep, to_return = buffer[split_point:], buffer[:split_point]
                    if to_return.strip():
                        yield to_return.strip()
                    buffer = to_keep
                elif "." in token:
                    is_post_period = True
                    tokens_since_period = 0

                if is_post_period and tokens_since_period > TOKENS_TO_GENERATE_PAST_PERIOD:
                    sentences = split_sentences(buffer)
                    if len(sentences) > 1:
                        yield " ".join(sentences[:-1])
                        buffer = sentences[-1]
                    is_post_period = False
                    tokens_since_period = 0
                else:
                    tokens_since_period += 1

            elif isinstance(token, FunctionFragment):
                function_name_buffer += token.name
                function_args_buffer += token.arguments
        to_return = buffer.strip()
        if to_return:
            yield to_return
        if function_name_buffer and get_functions:
            yield FunctionCall(name=function_name_buffer, arguments=function_args_buffer)
