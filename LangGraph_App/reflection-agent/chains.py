from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(dotenv_path=Path("../../.env"))
# print(os.getenv("OPENAI_API_KEY"))
# print(os.getenv("TAVILY_API_KEY"))

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique"
            " and recommendations for the user's tweet."
            " always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate best twitter post possible for the user's request."
            " If the user provides critiques, respond with a revised version of your previous attempts.",
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

