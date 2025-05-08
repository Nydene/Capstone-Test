import streamlit as st
from st_mic_recorder import mic_recorder
import numpy as np
import shap
import matplotlib.pyplot as plt
import random
import whisper
import tempfile

# Whisper model
asr_model = whisper.load_model("base")

# CRLA knowledge base
crla_knowledge_base = {
    "What is the main idea of the story?": "The main idea is that honesty is always the best policy.",
    "Why did the character feel guilty?": "Because they lied about breaking the vase.",
    "What lesson did the story teach?": "The story teaches that telling the truth builds trust.",
    "What did the mother do after learning the truth?": "She forgave the child and appreciated their honesty.",
    "How would you react if you were in the character's shoes?": "I would also tell the truth even if it's hard."
}

reading_passage = "A child broke a vase while playing indoors. At first, they hid the truth out of fear. Later, they told their mother, who forgave them and praised their honesty."

st.title("Reading Comprehension Analyzer")

if 'page' not in st.session_state:
    st.session_state.page = 'prompt'

if st.session_state.page == 'prompt':
    st.header("Step 1: Read the Passage")
    st.success(reading_passage)
    if st.button("I'm ready to retell"):
        st.session_state.page = 'retell'

elif st.session_state.page == 'retell':
    st.header("Step 2: Retell the Story")
    audio = mic_recorder(start_prompt="Click to start recording retelling", stop_prompt="Stop recording", key="retell_mic")
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio)
            tmp_path = tmp_file.name
        result = asr_model.transcribe(tmp_path)
        st.session_state.retelling_text = result['text']
        st.text_area("Transcribed Retelling:", value=st.session_state.retelling_text, height=100)
    if st.button("Proceed to Questions"):
        st.session_state.page = 'questions'

elif st.session_state.page == 'questions':
    st.header("Step 3: Answer the Questions")
    questions = list(crla_knowledge_base.keys())
    random.shuffle(questions)

    if 'user_scores' not in st.session_state:
        st.session_state.user_scores = []
        st.session_state.labels = []
        st.session_state.answered = [False] * 5
        st.session_state.q_data = questions[:5]

    for i in range(5):
        question = st.session_state.q_data[i]
        st.subheader(f"Question {i+1}: {question}")
        audio = mic_recorder(start_prompt="Record your answer", stop_prompt="Stop recording", key=f"mic_q{i}")
        if audio and not st.session_state.answered[i]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio)
                tmp_path = tmp_file.name
            result = asr_model.transcribe(tmp_path)
            answer_text = result['text']
            st.text_area(f"Transcribed Answer {i+1}", value=answer_text, key=f"ans_{i}")
            correct = crla_knowledge_base[question]
            score = 1.0 if answer_text.strip().lower() in correct.lower() else 0.0
            st.session_state.user_scores.append(score)
            st.session_state.labels.append(question)
            st.session_state.answered[i] = True

    if all(st.session_state.answered):
        final_score = sum(st.session_state.user_scores)
        st.success(f"Final Score: {final_score}/5")

        weak_areas = [st.session_state.labels[i] for i in range(5) if st.session_state.user_scores[i] == 0]
        if weak_areas:
            st.warning("Areas to Improve:")
            for w in weak_areas:
                st.write(f"- {w}")
        else:
            st.balloons()
            st.success("Excellent comprehension! 🎉")

        st.subheader("SHAP Feature Contribution")
        explainer = shap.Explainer(lambda x: np.array(st.session_state.user_scores))
        shap_values = explainer(np.zeros((1, len(st.session_state.user_scores))))
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
