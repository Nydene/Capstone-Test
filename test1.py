import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import shap
import matplotlib.pyplot as plt
import random
import os
import tempfile
import whisper

# Whisper ASR model (can use 'base' or 'small' for speed)
asr_model = whisper.load_model("base")

# Sample RAG-like knowledge base for CRLA-aligned answers (simplified)
crla_knowledge_base = {
    "What is the main idea of the story?": "The main idea is that honesty is always the best policy.",
    "Why did the character feel guilty?": "Because they lied about breaking the vase.",
    "What lesson did the story teach?": "The story teaches that telling the truth builds trust.",
    "What did the mother do after learning the truth?": "She forgave the child and appreciated their honesty.",
    "How would you react if you were in the character's shoes?": "I would also tell the truth even if it's hard."
}

reading_passage = "A child broke a vase while playing indoors. At first, they hid the truth out of fear. Later, they told their mother, who forgave them and praised their honesty."

# UI
st.title("Reading Comprehension Analyzer Prototype")

st.markdown("### Step 1: Read the Passage Aloud")
st.success(reading_passage)
st.markdown("(Use your mic to record)")

# Audio recording and transcription
st.markdown("### Step 2: Retell what you remember")
retelling_audio = st.file_uploader("Upload retelling audio", type=["wav", "mp3", "m4a"])
retelling_text = ""
if retelling_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(retelling_audio.read())
        tmp_path = tmp_file.name
    result = asr_model.transcribe(tmp_path)
    retelling_text = result["text"]
    st.text_area("Transcribed Retelling:", value=retelling_text, height=100)

# Step 3: Comprehension Questions
st.markdown("### Step 3: Answer Questions by Uploading Voice Responses")
questions = list(crla_knowledge_base.keys())
random.shuffle(questions)

user_scores = []
labels = []

for i in range(5):
    question = questions[i]
    st.markdown(f"**Q{i+1}: {question}**")
    audio = st.file_uploader(f"Upload your voice answer to question {i+1}", type=["wav", "mp3", "m4a"], key=f"audio_{i}")
    answer_text = ""
    if audio is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio.read())
            tmp_path = tmp_file.name
        result = asr_model.transcribe(tmp_path)
        answer_text = result["text"]
        st.text_area(f"Your Answer (transcribed):", value=answer_text, height=50, key=f"ans_{i}")
        correct = crla_knowledge_base[question]
        score = 1.0 if answer_text.strip().lower() in correct.lower() else 0.0
        user_scores.append(score)
        labels.append(question)

# Step 4: Feedback
if user_scores:
    final_score = sum(user_scores)
    st.success(f"Final Score: {final_score}/5")

    weak_areas = [labels[i] for i in range(len(user_scores)) if user_scores[i] == 0]
    if weak_areas:
        st.warning("Areas to Improve:")
        for w in weak_areas:
            st.write(f"- {w}")
    else:
        st.balloons()
        st.success("Excellent comprehension! 🎉")

    # SHAP Visualization
    st.subheader("SHAP Feature Contribution")
    explainer = shap.Explainer(lambda x: np.array(user_scores))
    shap_values = explainer(np.zeros((1, len(user_scores))))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')
