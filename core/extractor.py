"""
core/extractor.py
-----------------
Extracts structured learning insights from an educational video transcript:
  - Study Tasks (actionable things to do after watching)
  - Core Concepts (fundamental ideas introduced)
  - Open Questions (topics left unresolved or worth exploring further)
"""

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.base import RunnableSerializable
from config import Config, logger


def get_llm() -> ChatMistralAI:
    """Returns the configured Groq LLM instance for extraction tasks."""
    return ChatMistralAI(
        model="open-mistral-nemo",
        mistral_api_key=Config.MISTRAL_API_KEY,
        temperature=0.2,
    )


def build_chain(system_prompt: str) -> RunnableSerializable:
    """Builds a standard extraction chain given a system prompt."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """
    Extract concrete study tasks from the educational video transcript.

    Args:
        transcript (str): Full transcript text.

    Returns:
        str: Numbered list of study tasks.
    """
    logger.info("Extracting study tasks.")
    try:
        chain = build_chain(
            "You are an expert educational analyst. From this educational video transcript, "
            "extract specific STUDY TASKS the learner should do to reinforce and apply what they learned.\n\n"
            "For each task provide:\n"
            "- What to do (clear, actionable)\n"
            "- Why it helps (brief justification)\n\n"
            "Format as a numbered list. If none can be identified, say 'No specific study tasks identified.'"
        )
        return chain.invoke(transcript)
    except Exception as e:
        logger.error(f"Failed to extract study tasks: {e}")
        return "No study tasks found (Error extracting data)."


def extract_key_decisions(transcript: str) -> str:
    """
    Extract core concepts introduced in the educational video.

    Args:
        transcript (str): Full transcript text.

    Returns:
        str: Numbered list of core concepts.
    """
    logger.info("Extracting core concepts.")
    try:
        chain = build_chain(
            "You are an expert educational analyst. From this educational video transcript, "
            "extract the CORE CONCEPTS and fundamental ideas introduced.\n\n"
            "For each concept:\n"
            "- Name the concept clearly\n"
            "- Give a 1-sentence definition or explanation as presented in the video\n\n"
            "Format as a numbered list. If none can be identified, say 'No core concepts identified.'"
        )
        return chain.invoke(transcript)
    except Exception as e:
        logger.error(f"Failed to extract core concepts: {e}")
        return "No core concepts found (Error extracting data)."


def extract_questions(transcript: str) -> str:
    """
    Extract open questions and topics worth exploring further.

    Args:
        transcript (str): Full transcript text.

    Returns:
        str: Numbered list of open questions.
    """
    logger.info("Extracting open questions for further exploration.")
    try:
        chain = build_chain(
            "You are an expert educational analyst. From this educational video transcript, "
            "identify OPEN QUESTIONS and topics that were raised but not fully answered, "
            "or that a curious learner should investigate further.\n\n"
            "Format as a numbered list of clear, specific questions. "
            "If none can be identified, say 'No open questions identified.'"
        )
        return chain.invoke(transcript)
    except Exception as e:
        logger.error(f"Failed to extract open questions: {e}")
        return "No open questions found (Error extracting data)."