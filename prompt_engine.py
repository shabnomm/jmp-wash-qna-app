import os
import re
from langdetect import detect
from googletrans import Translator
from transformers import pipeline
from explanation_dict import EXPLANATIONS

# Initialize translator
translator = Translator()

# Multilingual QA model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# Cache for performance
qa_cache = {}

# Global variable for uploaded context text
uploaded_text = None

def translate_to_english(text):
    return translator.translate(text, src='auto', dest='en').text

def translate_to_bangla(text):
    return translator.translate(text, src='auto', dest='bn').text

def generate_cache_key(text):
    return text.lower().strip()

def get_best_context_chunks(long_text, question, max_len=512, top_n=2):
    lines = long_text.split('. ')
    question_keywords = set(question.lower().split())

    scored_lines = []
    for line in lines:
        line_words = set(line.lower().split())
        score = len(question_keywords & line_words)
        if score > 0:
            scored_lines.append((score, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    best_lines = [line for score, line in scored_lines[:top_n]]

    combined = ". ".join(best_lines)
    return combined[:max_len] if combined else long_text[:max_len]

def query_fallback_model(prompt):
    return "Sorry, I could not find a precise answer."

def get_glossary_explanation(question_en):
    """
    Return matched glossary explanation if any keyword matches as a whole word
    """
    for keyword, texts in EXPLANATIONS.items():
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, question_en.lower()):
            return texts.get("en", "")
    return ""

def prepare_context(question_en, context=None):
    global uploaded_text

    if context is not None:
        return context
    elif uploaded_text:
        lang = detect(uploaded_text)
        if lang == 'bn':
            text_context_en = translate_to_english(uploaded_text)
        else:
            text_context_en = uploaded_text
        return get_best_context_chunks(text_context_en, question_en)
    else:
        # Default fallback context
        return (
            "Clean water is water that is safe to drink and free from harmful contaminants. "
            "Sanitation refers to the provision of facilities and services for the safe disposal of human urine and feces."
        )

import re

def generate_answer(question, context=None):
    lang = detect(question)
    question_en = translate_to_english(question) if lang != 'en' else question
    cache_key = generate_cache_key(question_en)

    print(f"🧠 Detected language: {lang}")
    print(f"🔎 Question (EN): {question_en}")

    # Check glossary match first (top priority)
    for keyword, texts in EXPLANATIONS.items():
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, question_en.lower()):
            # Return glossary explanation exactly in the question's language
            if lang == 'bn' and 'bn' in texts:
                print("📚 Matched glossary keyword, returning Bangla explanation.")
                return texts['bn']
            else:
                print("📚 Matched glossary keyword, returning English explanation.")
                return texts.get('en', "Sorry, no explanation available.")

    # If no glossary match, check cache or run QA model as before
    if cache_key in qa_cache:
        print("🔁 Returning cached result")
        answer_en = qa_cache[cache_key]
    else:
        if context is None:
            if uploaded_text:
                text_context = get_best_context_chunks(uploaded_text, question_en)
            else:
                text_context = (
                    "Clean water is water that is safe to drink and free from harmful contaminants. "
                    "Sanitation refers to the provision of facilities and services for the safe disposal of human urine and feces."
                )
        else:
            text_context = context

        try:
            result = qa_pipeline(question=question_en, context=text_context)
            answer_en = result.get("answer", "").strip()

            if not answer_en or answer_en.lower() in ["", "no answer", "unknown"]:
                raise ValueError("Weak answer from QA model")

        except Exception as e:
            print(f"⚠️ QA failed: {e}")
            fallback_prompt = f"Answer the following question:\n\nQuestion: {question_en}\n\nContext: {text_context}"
            answer_en = query_fallback_model(fallback_prompt)

        qa_cache[cache_key] = answer_en

    # Translate answer to Bangla only if original question was not in English
    return translate_to_bangla(answer_en) if lang != 'en' else answer_en
