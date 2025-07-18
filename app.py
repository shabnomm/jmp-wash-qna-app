import streamlit as st
from prompt_engine import generate_answer
from text_splitter import get_context_from_text
import fitz  # PyMuPDF

# Set background and styles
def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Roboto', sans-serif;
        }}

        /* Global white text except inputs */
        html, body, [class*="css"]:not(input):not(textarea):not(.stTextInput input) {{
            color: #FFFFFF !important;
        }}

        /* Title and subtitle */
        h1, .stMarkdown > p {{
            color: #FFFFFF !important;
        }}

        /* File uploader label */
        .stFileUploader label {{
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        /* Text input label */
        .stTextInput label {{
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        /* Input text (typed question remains black) */
        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.15);
            color: #000000 !important;  /* keep text black */
            border-radius: 10px;
            padding: 10px;
        }}

        .monospace-center {{
            font-family: 'Lucida Console', monospace;
            color: #FFFFFF;
            text-align: center;
            font-size: 18px;
            margin-top: 10px;
            margin-bottom: 20px;
        }}

        /* Answer box */
        .answer-box {{
            background-color: rgba(17,24,39, 0.85);
            color: #FFFFFF;
            padding: 20px;
            margin-top: 20px;
            border-radius: 15px;
            font-size: 18px;
            font-family: 'Lucida Console', monospace;
            text-shadow: 1px 1px 2px #000;
        }}

        /* Label 'Answer:' */
        .monospace-label {{
            font-family: 'Lucida Console', monospace;
            font-size: 20px;
            color: #FFFFFF;
            margin-top: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background early
set_background("https://i.postimg.cc/j5Lzx0x8/20250718-1242-image.png")

# Extract text from PDF
def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# UI components
st.title("JMP Wash Bilingual QnA Assistant")

# Monospace, white, centered subtitle just below title
st.markdown(
    '<p class="monospace-center">Ask any sanitation or hygiene question in Bangla or English.</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📄 Upload PDF or Text file with JMP Wash info", type=["pdf", "txt"])

uploaded_text = None
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        uploaded_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        uploaded_text = uploaded_file.read().decode("utf-8")

question = st.text_input("💬 Enter your question here:")

if question:
    context = (
        get_context_from_text(uploaded_text, question)
        if uploaded_text
        else (
            "Clean water is water that is safe to drink and free from harmful contaminants. "
            "Sanitation refers to the provision of facilities and services for the safe disposal of human urine and feces."
        )
    )
    answer = generate_answer(question, context=context)
    st.markdown('<div class="monospace-label">Answer:</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

