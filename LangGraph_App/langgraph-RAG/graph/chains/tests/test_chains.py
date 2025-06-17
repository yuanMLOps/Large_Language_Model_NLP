from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

import os

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

from graph.chains.generation import generation_chain
from pprint import pprint
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import question_router, RouteQuery


# print(os.getenv("OPENAI_API_KEY"))


def test_retrival_grader_answer_yes() -> None:

    question = "generative agents"
    docs = retriever.invoke(question)


    doc_txt = docs[0].page_content
    print(doc_txt)

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:

    question = "generative agents"
    docs = retriever.invoke(question)


    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "generative agents"
    doc = retriever.invoke(question)
    generation = generation_chain.invoke({"context": doc, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "generative agents"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "generative agents"
    docs = retriever.invoke(question)

    generation = "In order to make pizza we need to first start with the dough"
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "generative agents"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_vectorstore() -> None:
    question = "how to become a millionaire?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

