import re

def clean_text(text):
    """Basic cleanup: remove extra spaces and line breaks."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for better context coverage.
    
    Args:
      text (str): the full text
      chunk_size (int): max chars per chunk
      overlap (int): number of chars to overlap between chunks
      
    Returns:
      List[str]: list of text chunks
    """
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move start with overlap
    return chunks

def get_context_from_text(text, question, chunk_size=500, top_k=3):
    """
    Return top K chunks most relevant to the question.
    
    Uses simple keyword matching score.
    
    Args:
      text (str): full text
      question (str): question in English
      chunk_size (int): chunk size for splitting
      top_k (int): number of top chunks to return
    
    Returns:
      str: concatenated top-k relevant chunks
    """
    chunks = chunk_text(text, chunk_size)
    question_words = set(question.lower().split())
    
    def score(chunk):
        chunk_words = set(chunk.lower().split())
        return len(question_words.intersection(chunk_words))
    
    scored_chunks = [(score(chunk), chunk) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top k chunks with highest keyword match
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    
    return " ".join(top_chunks)
