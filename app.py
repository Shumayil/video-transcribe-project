from dotenv import load_dotenv
import os
import streamlit as st
# make sure you’ve created transcribe.py, indexer.py, qa.py with the functions we'll call

# 1. load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. basic Streamlit UI
st.title("Lecture Q&A")

# Upload widget
uploaded = st.file_uploader("Upload lecture video or audio", type=["mp4","mp3","wav"])
if uploaded:
    # save to disk
    path = os.path.join("lectures", uploaded.name)
    os.makedirs("lectures", exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {path}")

    # call your transcription function (you’ll write this in transcribe.py)
    from transcribe import transcribe_file
    transcript_path = transcribe_file(path)

    st.success(f"Transcript saved at {transcript_path}")

# question input
question = st.text_input("Ask a question about the lecture")
if st.button("Get Answer"):
    from indexer import query_index
    from qa import answer_question
    # 1) semantic search
    chunks = query_index(question, k=5)
    # 2) ask GPT
    answer = answer_question(question, "\n---\n".join(chunks), OPENAI_API_KEY)
    st.subheader("Answer")
    st.write(answer)
