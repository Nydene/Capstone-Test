import streamlit as st
import speech_recognition as sr
import random
import shap
import matplotlib.pyplot as plt
import numpy as np

# Sample RAG-like knowledge base for CRLA-aligned answers (simplified)
crla_knowledge_base = {
    "What is the main idea of the story?": "The main idea is that honesty is always the best policy.",
    "Why did the character feel guilty?": "Because they lied about breaking the vase.",
    "What lesson did the story teach?": "The story teaches that telling the truth builds trust.",
    "What did the mother do after learning the truth?": "She forgave the child and appreciated their honesty.",
    "How would you react if you were in the character's shoes?": "I would also tell the truth even if it's hard."
}

# Define a sample reading passage
reading_passage = "A child broke a vase while playing indoors. At first, they hid the truth out of fear. Later, they told their mother, who forgave them and praised their honesty."

# Simulated STT (replace with real speech recognition logic for full deployment)
def simulate_speech_input(prompt):
    st.info(f"Prompt: {prompt}")
    return st.text_input("Your spoken answer (simulated)", "")

# Scoring logic based on matching to knowledge base

def score_answer(user_answer, correct_answer):
    score = 1.0
    if user_answer.strip().lower() in correct_answer.lower():
        return score
    return 0.0

# SHAP explanation for feedback

def generate_shap_feedback(scores, labels):
    explainer = shap.Explainer(lambda x: np.array(scores))
    shap_values = explainer(np.zeros((1, len(scores))))

    st.subheader("SHAP Feature Contribution")
    fig = shap.plots.bar(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')

# Streamlit UI
st.title("Reading Comprehension Analyzer Prototype")

st.markdown("### Step 1: Read the Passage Aloud")
st.success(reading_passage)
st.markdown("(Use your mic and read aloud — currently simulated)")

# Step 2: Retelling
retelling = simulate_speech_input("Now, retell what you remember.")

# Step 3: Comprehension Questions
st.markdown("### Step 3: Answer Questions")
questions = list(crla_knowledge_base.keys())
random.shuffle(questions)

user_scores = []
labels = []

for i in range(5):
    question = questions[i]
    correct = crla_knowledge_base[question]
    user_response = simulate_speech_input(question)
    score = score_answer(user_response, correct)
    user_scores.append(score)
    labels.append(question)

# Step 4: Final Feedback
final_score = sum(user_scores)
st.success(f"Final Score: {final_score}/5")

# Identify Weak Areas
weak_areas = [labels[i] for i in range(len(user_scores)) if user_scores[i] == 0]
if weak_areas:
    st.warning("Areas to Improve:")
    for w in weak_areas:
        st.write(f"- {w}")
else:
    st.balloons()
    st.success("Excellent comprehension! 🎉")

# Step 5: SHAP Analysis
generate_shap_feedback(user_scores, labels)
