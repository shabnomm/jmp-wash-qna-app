import os
import re
import requests
from langdetect import detect
from googletrans import Translator
from transformers import pipeline
from explanation_dict import EXPLANATIONS

# Initialize translator
translator = Translator()

# QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

qa_cache = {}
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

    if len(combined) < max_len // 2:
        combined = long_text[:max_len]

    return combined[:max_len]

def query_llm_fallback(prompt):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    headers = {"Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.5, "max_new_tokens": 150}
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        response = r.json()
        return response[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        print("⚠️ LLM fallback failed:", e)
        return "Sorry, I could not find a precise answer."

def prepare_context(question_en, context=None):
    global uploaded_text

    if context:
        return context
    elif uploaded_text:
        lang = detect(uploaded_text)
        if lang == 'bn':
            text_context_en = translate_to_english(uploaded_text)
        else:
            text_context_en = uploaded_text
        return get_best_context_chunks(text_context_en, question_en)
    else:
        return (
            "Clean water is water that is safe to drink and free from harmful contaminants. "
            "Sanitation refers to the provision of facilities and services for the safe disposal of human urine and feces."
        )

def generate_answer(question, context=None):
    lang = detect(question)
    question_en = translate_to_english(question) if lang != 'en' else question
    cache_key = generate_cache_key(question_en)

    print(f"🧠 Detected language: {lang}")
    print(f"🔎 Question (EN): {question_en}")

    # ✅ 1. Glossary answer first
    for keyword, texts in EXPLANATIONS.items():
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, question_en.lower()):
            print("📚 Matched glossary keyword.")
            if lang == 'bn' and 'bn' in texts:
                return texts['bn']
            else:
                return texts.get('en', "No explanation available.")

    # ✅ 2. Cache
    if cache_key in qa_cache:
        print("🔁 Returning cached result")
        answer_en = qa_cache[cache_key]
    else:
        text_context = prepare_context(question_en, context)

        try:
            result = qa_pipeline(question=question_en, context=text_context)
            answer_en = result.get("answer", "").strip()

            if not answer_en or answer_en.lower() in ["", "no answer", "unknown"]:
                raise ValueError("Weak answer from QA model")

        except Exception as e:
            print(f"⚠️ QA failed: {e}")
            fallback_prompt = f"Question: {question_en}\nContext: {text_context}\nAnswer:"
            answer_en = query_llm_fallback(fallback_prompt)

        qa_cache[cache_key] = answer_en

    return translate_to_bangla(answer_en) if lang != 'en' else answer_en
