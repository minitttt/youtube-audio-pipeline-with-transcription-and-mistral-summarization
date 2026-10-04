from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.base import RunnableSerializable
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from config import Config, logger

def get_llm() -> ChatMistralAI:
    """Returns the configured Groq LLM for RAG."""
    return ChatMistralAI(
        model="open-mistral-nemo",
        mistral_api_key=Config.MISTRAL_API_KEY,
        temperature=0.3,
    )

def format_docs(docs: list) -> str:
    """Formats retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript: str) -> RunnableSerializable:
    """
    Builds the complete RAG chain for asking questions.
    
    Args:
        transcript (str): The transcript text to build the index from.
        
    Returns:
        RunnableSerializable: The LangChain RAG chain.
    """
    logger.info("Building RAG chain.")
    try:
        vector_store = build_vector_store(transcript)
        retriever = get_retriever(vector_store, k=10)
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert tutor and educational assistant. A student has just watched an educational video and is asking you questions about it.

Answer their question based ONLY on the video transcript context provided below. Explain clearly, use specific examples from the transcript, and help the student truly understand — not just recall facts.

If the answer is not in the transcript, say:
"This topic wasn't covered in the video. I'd recommend looking it up separately."

Always be encouraging, precise, and pedagogically useful.

Context from the video transcript:
{context}""",
            ),
            ("human", "{question}"),
        ])

        rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt 
            | llm 
            | StrOutputParser()
        )

        logger.info("RAG chain successfully built.")
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to build RAG chain: {e}")
        raise

def load_rag_chain() -> RunnableSerializable:
    """Loads an existing RAG chain without rebuilding the vector store."""
    logger.info("Loading existing RAG chain.")
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=10)
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain: RunnableSerializable, question: str) -> str:
    """
    Ask a question using the RAG chain.
    
    Args:
        rag_chain (RunnableSerializable): The RAG chain.
        question (str): The user's question.
        
    Returns:
        str: The answer.
    """
    logger.info(f"User Question: {question}")
    try:
        answer = rag_chain.invoke(question)
        logger.info(f"Assistant Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error querying RAG chain: {e}")
        return "Sorry, I encountered an error while trying to answer your question."