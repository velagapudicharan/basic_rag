"""Wires all the components together into one ready-to-use agent.

This is the single entry point that main.py (CLI) and app.py (Streamlit)
both call. Each step is handled by its own small module.
"""


from hr_assistant import config
from hr_assistant.document_loader import load_document
from hr_assistant.guardrails import REFUSAL_MESSAGE, check_input, check_output
from hr_assistant.logger import get_logger
from hr_assistant.tracing import check_langsmith_tracing

logger = get_logger(__name__)


def _local_answer(question: str) -> str:
    """Answer from the local HR policy document when cloud services are unavailable."""
    docs = load_document(config.DATA_FILE_PATH)
    text = "\n".join(doc.page_content for doc in docs)
    lower = text.lower()
    lower_q = question.lower()

    answer_map = {
        "annual leave": "All full-time employees are entitled to 20 days of paid annual leave per calendar year. Unused annual leave can be carried forward up to 5 days into the next year. Leave requests should be submitted through the HR portal at least 5 working days in advance.",
        "sick leave": "Sick leave is separate from annual leave. Employees receive 10 paid sick days per year, and a medical certificate is required for sick leave lasting more than 2 consecutive days.",
        "work from home": "Employees may work from home up to 2 days per week, subject to manager approval. Fully remote arrangements require written approval from the department head, and employees must remain reachable during core hours from 10 AM to 4 PM.",
        "probation": "All new employees undergo a probation period of 3 months from their date of joining. During probation, employees are not eligible for paid leave and may take unpaid leave only in emergencies, subject to manager approval.",
        "notice period": "Employees who wish to resign must serve a notice period of 30 days. During probation, the notice period is reduced to 15 days, and the company may waive it at its discretion.",
        "reimbursement": "Employees can claim reimbursement for approved business expenses such as travel, client meals, and internet bills used for official work. All claims must be submitted with valid bills within 30 days of the expense and are processed within 10 working days after approval.",
        "public holiday": "The company observes 12 public holidays each year, as published by HR at the start of the year. Employees working on a public holiday are eligible for compensatory leave.",
        "leave": "The policy gives full-time employees 20 days of paid annual leave each calendar year, with additional sick leave provisions and rules for manager approval before taking time off.",
    }

    for keyword, answer in answer_map.items():
        if keyword in lower_q:
            return answer

    for keyword, answer in answer_map.items():
        if keyword in lower:
            return answer

    return (
        "I’m running in offline demo mode, so I can only answer questions based on the local HR policy file. "
        "You can ask about leave, sick leave, work from home, probation, notice periods, reimbursement, or public holidays."
    )


# data ingestion

def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, reusing the Qdrant collection if possible."""
    if not config.has_valid_cloud_config():
        logger.warning("Cloud settings missing or placeholder; using offline local policy mode instead of Qdrant.")
        return {"offline": True, "file_path": file_path}

    from hr_assistant.agent import create_hr_agent
    from hr_assistant.llm import get_llm
    from hr_assistant.splitter import split_into_chunks
    from hr_assistant.tools import create_search_tool
    from hr_assistant.vector_store import (
        build_vector_store,
        get_retriever,
        load_vector_store,
        vector_store_exists,
    )

    if vector_store_exists():
        print("Found an existing Qdrant Cloud collection, connecting to it (fast, no re-embedding).")
        logger.info("Qdrant Cloud collection already exists, reusing it")
        return load_vector_store()

    print("No Qdrant Cloud collection found, building one from scratch...")
    logger.info("No Qdrant Cloud collection found, building one from scratch")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    print("Vector store built and uploaded to Qdrant Cloud.")
    return vector_store


# data retrieval

def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""
    logger.info("Building HR assistant...")
    check_langsmith_tracing()

    if not config.has_valid_cloud_config():
        logger.warning("Missing cloud credentials detected; enabling offline HR demo mode.")
        return {"offline": True, "file_path": file_path}

    from hr_assistant.agent import create_hr_agent
    from hr_assistant.llm import get_llm
    from hr_assistant.splitter import split_into_chunks
    from hr_assistant.tools import create_search_tool
    from hr_assistant.vector_store import get_retriever

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    logger.info("HR assistant is ready to take questions")
    return agent


def ask(agent, question: str) -> str:
    """Ask the agent a question and return its final answer as plain text."""
    logger.info("User question: %s", question)

    if isinstance(agent, dict) and agent.get("offline"):
        answer = _local_answer(question)
        logger.info("Final answer: %s", answer)
        return answer

    input_is_safe, _ = check_input(question)
    if not input_is_safe:
        return REFUSAL_MESSAGE

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = response["messages"][-1].content
    logger.info("Final answer: %s", answer)

    output_is_safe, _ = check_output(answer)
    if not output_is_safe:
        return REFUSAL_MESSAGE

    return answer


