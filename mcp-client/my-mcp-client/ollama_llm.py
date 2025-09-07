import subprocess
import json
from click import prompt
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class OllamaLLM(Runnable):
    def __init__(self, model_name="llama2", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, inputs: ChatPromptTemplate, run_manager=None, **kwargs) -> dict:
        messages = inputs.to_messages()
        prompt_text = "\n".join(
             f"{m.type.upper()}: {m.content}" for m in messages
        )

        proc = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = proc.communicate(input=(prompt_text + "\n").encode("utf-8"), timeout=30)

        if proc.returncode != 0:
            raise RuntimeError(f"Ollama call failed: {stderr}")

        return [{"role": "assistant", "content": stdout.decode("utf-8").strip()}]
        

    def invoke_async(self, inputs: dict, run_manager=None, **kwargs) -> dict:
        messages = inputs.to_messages()
        prompt_text = "\n".join(
             f"{m.type.upper()}: {m.content}" for m in messages
        )

        proc = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = proc.communicate(input=(prompt_text + "\n").encode("utf-8"), timeout=30)

        if proc.returncode != 0:
            raise RuntimeError(f"Ollama call failed: {stderr}")

        return {
            [{"role": "assistant", "content": stdout.decode("utf-8").strip()}]
        }

    def bind_tools(self, tools):
        return self

    def __or__(self, other):
        return self
