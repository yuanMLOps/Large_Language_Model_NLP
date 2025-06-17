from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    """grade if the answer addresses the question """

    binary_score: bool = Field(
        description="answer addresses the question, yes or no"
    )

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system="""You are a grader assessing whether an LLM generation addresses/resolves a question \n
  Give a binary score of 'yes' or 'no'. 'Yes' means that the LLM generation resolves the question
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

