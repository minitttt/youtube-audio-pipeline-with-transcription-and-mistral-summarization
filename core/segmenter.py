"""
core/segmenter.py
-----------------
Splits a video transcript into labeled topic segments / learning chapters
using the Mistral LLM. Each segment has a title and a short description.
"""

import json
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import Config, logger

_SEGMENT_SYSTEM_PROMPT = """You are an expert educational content analyst.

Given the full transcript of an educational video (lecture, tutorial, talk, etc.),
identify the distinct TOPIC SEGMENTS / CHAPTERS covered in the video.

For each segment return a JSON array. Each element must have exactly these keys:
  "chapter_number" : integer starting from 1
  "title"          : short chapter title (max 8 words)
  "description"    : 2–3 sentence summary of what this chapter covers

Return ONLY the raw JSON array — no markdown, no code fences, no extra text.

Example output:
[
  {{"chapter_number": 1, "title": "Introduction to Neural Networks", "description": "The speaker introduces the concept of artificial neurons and biological inspiration. Key terms like weights, biases and activation functions are defined."}},
  {{"chapter_number": 2, "title": "Backpropagation Explained", "description": "A step-by-step walkthrough of the backpropagation algorithm. The chain rule is applied to compute gradients and update weights."}}
]"""


def segment_transcript(transcript: str) -> list[dict]:
    """
    Splits the transcript into topic segments using Mistral LLM.

    Args:
        transcript (str): Full transcript text.

    Returns:
        list[dict]: A list of segment dicts with keys:
                    chapter_number, title, description.
    """
    logger.info("Starting topic segmentation.")
    try:
        llm = ChatMistralAI(
            model="open-mistral-nemo",
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0.2,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SEGMENT_SYSTEM_PROMPT),
            ("human", "Here is the transcript to segment:\n\n{text}"),
        ])

        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | prompt
            | llm
            | StrOutputParser()
        )

        # Use first ~6000 chars — sufficient to detect topic boundaries
        raw = chain.invoke(transcript[:6000]).strip()
        logger.debug(f"Raw segmentation output: {raw[:300]}")

        # Clean markdown code blocks if LLM included them
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        segments = json.loads(raw)
        logger.info(f"Identified {len(segments)} topic segments.")
        return segments

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse segmentation JSON: {e}")
        return [{"chapter_number": 1, "title": "Full Lecture", "description": "Could not auto-segment this transcript."}]
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return [{"chapter_number": 1, "title": "Full Lecture", "description": "Topic segmentation encountered an error."}]
