"""
core/followup.py
-----------------
Generates a structured study follow-up document from a video transcript:
  - Key Takeaways
  - Flashcard Q&A
  - Suggested Next Steps
  - Related Concepts / Prerequisites
"""

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import Config, logger

_FOLLOWUP_SYSTEM_PROMPT = """You are an expert educator and curriculum designer.

Given the transcript of an educational video, produce a complete STUDY FOLLOW-UP DOCUMENT.

Format your response using exactly the following structure (use these exact headings):

## 📌 Key Takeaways
List the top 5 most important things the student should remember from this video.
Use bullet points. Be concise and precise.

## 🃏 Flashcard Q&A
Generate 5 self-test questions with answers to help the student review the material.
Format each as:
**Q:** [Question]
**A:** [Answer]

## 📚 Suggested Next Steps
What should the student study or do next after watching this video?
Provide 3–5 concrete recommendations (topics, skills, practice tasks).

## 🔗 Prerequisites & Related Concepts
What prior knowledge does this video assume?
List any related concepts that would deepen understanding.

Be educational, accurate, and based strictly on the content of the transcript."""


def generate_study_followup(transcript: str) -> str:
    """
    Generates a structured study follow-up document.

    Args:
        transcript (str): Full transcript text.

    Returns:
        str: Markdown-formatted study follow-up document.
    """
    logger.info("Generating study follow-up document.")
    try:
        llm = ChatMistralAI(
            model="open-mistral-nemo",
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0.3,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", _FOLLOWUP_SYSTEM_PROMPT),
            ("human", "Here is the transcript:\n\n{text}"),
        ])

        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(transcript)
        logger.info("Study follow-up document generated successfully.")
        return result

    except Exception as e:
        logger.error(f"Failed to generate study follow-up: {e}")
        return "## Study Follow-Up\n\nCould not generate a study plan due to an error. Please try again."
