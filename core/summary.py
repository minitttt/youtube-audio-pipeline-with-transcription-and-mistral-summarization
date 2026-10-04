"""
core/summary.py
---------------
Generates a learning-focused summary and title for an educational video transcript
using Mistral LLM via a Map-Reduce pattern for long content.
"""

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import Config, logger


def get_llm() -> ChatMistralAI:
    """Returns the configured Groq LLM instance."""
    return ChatMistralAI(
        model="open-mistral-nemo",
        mistral_api_key=Config.MISTRAL_API_KEY,
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list[str]:
    """Splits a long transcript into chunks for LLM processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """
    Generate a learning-focused summary from an educational video transcript.
    Uses a Map-Reduce approach for long transcripts.

    Args:
        transcript (str): The full transcript text.

    Returns:
        str: A structured lesson summary in bullet points.
    """
    logger.info("Starting lesson summarization.")
    try:
        llm = get_llm()

        map_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a concise educational summarizer. "
                "Summarize this portion of an educational video transcript, "
                "focusing on the key concepts and ideas taught."
            ),
            ("human", "{text}"),
        ])
        map_chain = map_prompt | llm | StrOutputParser()

        chunks = split_transcript(transcript)
        logger.info(f"Split transcript into {len(chunks)} chunks for summarization.")

        chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
        combined = "\n\n".join(chunk_summaries)

        combined_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert educator. Combine these partial summaries of an educational video "
                "into one final, well-structured LESSON SUMMARY. "
                "Use bullet points. Focus on: what was taught, why it matters, and key insights. "
                "Write it so a student reviewing their notes would find it immediately useful."
            ),
            ("human", "{text}"),
        ])

        combined_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | combined_prompt
            | llm
            | StrOutputParser()
        )

        final_summary = combined_chain.invoke(combined)
        logger.info("Summarization complete.")
        return final_summary

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Summary could not be generated due to an error."


def generate_title(transcript: str) -> str:
    """
    Generate a short descriptive title for the educational video.

    Args:
        transcript (str): The transcript text (first 2000 chars used).

    Returns:
        str: The generated lesson title.
    """
    logger.info("Generating lesson title.")
    try:
        llm = get_llm()
        title_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Based on this educational video transcript, generate a clear and descriptive "
                    "LESSON TITLE (max 8 words). It should sound like a course chapter heading. "
                    "Return only the title — no quotes, no punctuation at the end."
                ),
                ("human", "{text}"),
            ])
            | llm
            | StrOutputParser()
        )
        title = title_chain.invoke(transcript[:2000])
        logger.info(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.error(f"Failed to generate title: {e}")
        return "Learning Video"
