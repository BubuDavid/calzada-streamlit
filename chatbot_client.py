import time

from transformers import pipeline


class Client:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")

    def chat(self, message: str):
        generated = self.generator(
            message, max_length=50, do_sample=True, temperature=0.7
        )
        response_message = generated[0]["generated_text"]
        response_message = response_message.replace(message, "")
        response_message = response_message.strip()
        return self.create_stream(response_message)

    def create_stream(self, message: str | list[str]):
        if isinstance(message, str):
            message = message.split()
            message = " ".join(message)
            for word in message.split():
                yield word + " "
                time.sleep(0.1)

        if isinstance(message, list):
            for line in message:
                for word in line.split():
                    yield word + " "
                    time.sleep(0.02)
