"""
main.py
-------
CLI entry point for the AI Learning Assistant pipeline.
"""

import sys
from utils.audio_processor import process_input, cleanup_files
from core.transcribe import transcribe_all
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.segmenter import segment_transcript
from core.followup import generate_study_followup
from core.rag_engine import build_rag_chain, ask_question
from config import logger


def run_pipeline(source: str, language: str = "english") -> dict:
    """
    Runs the full AI Learning Assistant pipeline.

    Args:
        source (str): YouTube URL or local file path.
        language (str): Language for transcription.

    Returns:
        dict: Results including title, transcript, summary, segments,
              study tasks, concepts, questions, followup, and rag_chain.
    """
    logger.info("Starting AI Learning Assistant Pipeline")

    try:
        # Try fetching transcript directly first to bypass yt-dlp blocks
        transcript = ""
        yt_id = None
        
        # Simple extraction for youtube id
        import re
        for pat in [
            r'(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})',
            r'(?:youtu\.be/)([A-Za-z0-9_-]{11})',
            r'(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})',
        ]:
            m = re.search(pat, source)
            if m: 
                yt_id = m.group(1)
                break

        if yt_id:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                api = YouTubeTranscriptApi()
                t_list = api.list(yt_id)
                try:
                    t = t_list.find_transcript(['en', 'en-US'])
                except Exception:
                    try:
                        t = t_list.find_generated_transcript(['en'])
                    except Exception:
                        t = next(iter(t_list))
                
                if language.lower() != "english" and t.is_translatable:
                    try: t = t.translate('en')
                    except: pass
                        
                transcript = " ".join([i.text for i in t.fetch()])
                logger.info("Successfully fetched transcript directly from YouTube!")
                audio_data = {}
                chunks = []
            except Exception as e:
                logger.warning(f"Failed to fetch direct transcript, falling back to audio download: {e}")

        if not transcript:
            audio_data = process_input(source)
            chunks = audio_data.get("chunks", [])
            transcript = transcribe_all(chunks, translate=(language.lower() != "english"))
        
        logger.info(f"Transcription preview: {transcript[:100]}...")

        title = generate_title(transcript)
        summary = summarize(transcript)
        study_tasks = extract_action_items(transcript)
        concepts = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        segments = segment_transcript(transcript)
        followup = generate_study_followup(transcript)
        rag_chain = build_rag_chain(transcript)

        # Cleanup temporary audio files
        cleanup_files(chunks)
        if "original_wav" in audio_data:
            cleanup_files([audio_data["original_wav"]])

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "study_tasks": study_tasks,
            "core_concepts": concepts,
            "open_questions": questions,
            "segments": segments,
            "study_followup": followup,
            "rag_chain": rag_chain,
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("🎓 AI Learning Assistant")
    try:
        source = input("Enter YouTube URL or local file path: ").strip()
        if not source:
            print("No source provided. Exiting.")
            sys.exit(1)

        language = input("Language (english/hinglish) [default: english]: ").strip() or "english"
        result = run_pipeline(source, language)

        print("\n" + "=" * 70)
        print(f"📖 Lesson Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n📚 Topic Segments:")
        for seg in result["segments"]:
            print(f"  Chapter {seg['chapter_number']}: {seg['title']}")
            print(f"    {seg['description']}")
        print(f"\n✅ Study Tasks:\n{result['study_tasks']}")
        print(f"\n💡 Core Concepts:\n{result['core_concepts']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print(f"\n📄 Study Follow-Up:\n{result['study_followup']}")
        print("=" * 70)

        print("\n💬 Ask the Video (type 'exit' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Keep learning!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🎓 Tutor: {answer}\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)