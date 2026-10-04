"""
core/infographic.py
-------------------
Generates Mermaid.js diagrams from transcript concepts using Mistral LLM.
"""

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import Config, logger

_MERMAID_PROMPT = """You are an expert educational content designer.
Your goal is to create a visual infographic (Mermaid.js code) based on the core concepts of an educational video.

Given a summary and core concepts, generate a Mermaid.js diagram. You can choose to generate either a mindmap or a flowchart (graph TD or graph LR). 

Follow these strict rules:
1. ONLY return the raw Mermaid.js code.
2. DO NOT include markdown code fences (like ```mermaid or ```).
3. DO NOT include any introductory or concluding text.
4. Keep the diagram concise and easy to read.
5. VERY IMPORTANT: Do NOT use parentheses `()`, brackets `[]`, or quotes `"` inside node labels unless you wrap the ENTIRE label in quotes, e.g., A["This is a (label)"]
6. Avoid special characters like : and { } in labels.

Here are the concepts and summary:
{text}
"""

def generate_infographic(summary: str, core_concepts: str) -> str:
    """
    Generates Mermaid.js code based on the summary and concepts.
    """
    logger.info("Starting infographic generation.")
    try:
        llm = ChatMistralAI(
            model="open-mistral-nemo",
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0.3,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", _MERMAID_PROMPT),
            ("human", "Generate the Mermaid graph for this:\n{text}"),
        ])

        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | prompt
            | llm
            | StrOutputParser()
        )

        input_text = f"SUMMARY:\n{summary}\n\nCONCEPTS:\n{core_concepts}"
        raw = chain.invoke(input_text).strip()
        
        # Clean markdown code blocks if the LLM still included them
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            
        # Clean "mermaid" if it's the first word due to stripping
        if raw.lower().startswith("mermaid"):
            raw = raw[7:].strip()

        logger.info("Infographic generation complete.")
        return raw

    except Exception as e:
        logger.error(f"Infographic generation failed: {e}")
        return "graph TD\n    A[Error] --> B[Failed to generate infographic]"
