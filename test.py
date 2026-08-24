# from dotenv import load_dotenv
# load_dotenv()   # MUST be before any core/ imports

# from utlis.audio_processor import process_input
# from core.transcribe import transcribe_all


# source = "https://www.youtube.com/watch?v=cTGqZ1gTM-4"  # Example YouTube URL
# chunks = process_input(source)      
# print(transcribe_all(chunks, translate=False))



from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports

from utlis.audio_processor import process_input
from core.transcribe import transcribe_all
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions



source = "https://www.youtube.com/watch?v=cTGqZ1gTM-4"  # Example YouTube URL
  # "english" → Whisper, "hinglish" → Sarvam



chunks = process_input(source)


transcript = transcribe_all(chunks, translate=False)  # Set translate=True for Hinglish to English translation
print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


title = generate_title(transcript)
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"📌 TITLE: {title}")
print("=" * 60)
print("\n📋 SUMMARY")
print("-" * 60)
print(summary)



action_items = extract_action_items(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 60)
print("✅ ACTION ITEMS")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("🔑 KEY DECISIONS")
print("=" * 60)
print(decisions)

print("\n" + "=" * 60)
print("❓ OPEN QUESTIONS")
print("=" * 60)
print(questions)