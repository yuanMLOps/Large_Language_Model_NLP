from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path("../../.env"))
print("hello world")

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    # print(app.invoke(input={"question": "how to make a pizza?"}))
    print(app.invoke(input={"question": "what is generative agents?"}))

