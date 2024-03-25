# Corrected import statements
import inspect
from typing import Any, Dict, Optional

# Corrected import path for RunnableConfig
from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.chains.base import Chain
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables import RunnableConfig, ensure_config


class CustomAgentExecutor(AgentExecutor):
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        intermediate_steps = []  # Initialize the list to capture intermediate steps

        # Ensure the configuration is set up correctly
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name")
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        # Prepare inputs based on the provided input
        inputs = self.prep_inputs(input)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        # Check if the _call method supports the new argument 'run_manager'
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            name=run_name,
        )

        # Capture the start of the chain as an intermediate step
        intermediate_steps.append(
            {"event": "Chain Started", "details": "Inputs prepared"}
        )

        try:
            # Execute the _call method, passing 'run_manager' if supported
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
            # Capture a successful call as an intermediate step
            intermediate_steps.append({"event": "Call Successful", "outputs": outputs})
        except BaseException as e:
            # Handle errors and capture them as intermediate steps
            run_manager.on_chain_error(e)
            intermediate_steps.append({"event": "Error", "error": str(e)})
            raise e
        finally:
            # Mark the end of the chain execution
            run_manager.on_chain_end(outputs)

        # Prepare the final outputs, including run information if requested
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs["run_info"] = RunInfo(run_id=run_manager.run_id)

        # Include intermediate steps in the final outputs
        final_outputs["intermediate_steps"] = intermediate_steps

        return final_outputs


if __name__ == "__main__":
    agent = CustomAgentExecutor()
