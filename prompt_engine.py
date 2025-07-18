import os
from langdetect import detect
from googletrans import Translator
from transformers import pipeline

# Initialize translator
translator = Translator()

# Initialize local QA pipeline (ensure model downloaded or auto-downloads)
local_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Cache dictionary for answered questions
qa_cache = {}

# Global variable for uploaded large text context (update this from your app if needed)
uploaded_text = None

def translate_to_english(text):
    """Translate input text to English"""
    return translator.translate(text, src='auto', dest='en').text

def translate_to_bangla(text):
    """Translate input text to Bangla"""
    return translator.translate(text, src='auto', dest='bn').text

def generate_cache_key(text):
    """Generate a simple cache key from the question text"""
    return text.lower().strip()

def get_context_from_text(long_text, question, max_len=512):
    """
    Extract context chunk from long text.
    This is a simple approach returning the first max_len chars.
    You can improve with semantic search or chunking later.
    """
    # TODO: Improve with semantic similarity search if desired
    return long_text[:max_len]

def query_fallback_model(prompt):
    """
    Placeholder for fallback LLM call.
    Implement external API call here (HuggingFace, OpenAI, etc.)
    For now, returns a fixed response.
    """
    return "Sorry, I could not find a precise answer."

def generate_answer(question, context=None):
    """
    Main function to generate answer for a question.
    Uses local QA model first, falls back to external LLM.
    Handles translation between Bangla and English.
    """
    lang = detect(question)
    
    question_en = translate_to_english(question) if lang != 'en' else question
    cache_key = generate_cache_key(question_en)

    if cache_key in qa_cache:
        print("🔁 Returning cached result")
        answer_en = qa_cache[cache_key]
    else:
        # Build or use provided context
        if context is None:
            if uploaded_text:
                context = get_context_from_text(uploaded_text, question_en)
            else:
                context = (
                    "Clean water is water that is safe to drink and free from harmful contaminants. "
                    "Sanitation refers to the provision of facilities and services for the safe disposal of human urine and feces."
                )
        try:
            # Query local QA pipeline
            result = local_qa(question=question_en, context=context)
            answer_en = result.get("answer", "").strip()
            # Fallback if local answer is weak or empty
            if not answer_en or answer_en.lower() in ["", "no answer", "unknown"]:
                raise ValueError("Local model returned weak answer.")
        except Exception as e:
            print(f"⚠️ Local model failed: {e}")
            prompt = f"Answer the following question clearly and concisely:\n\nQuestion: {question_en}\n\nContext: {context}\n\nAnswer:"
            answer_en = query_fallback_model(prompt)

        qa_cache[cache_key] = answer_en

    # Translate back if original question was not English
    if lang != 'en':
        return translate_to_bangla(answer_en)
    else:
        return answer_en
