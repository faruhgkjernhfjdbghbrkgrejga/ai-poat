import streamlit as st
import openai
import json

def get_explanation(quiz, correct_answer):
    prompt = f"문제: {quiz}\n정답: {correct_answer}\n이 문제의 해설을 작성해 주세요."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    explanation = response.choices[0].text.strip()
    return explanation

def quiz_review_page():
    st.title("퀴즈 리뷰 페이지")
    st.markdown("---")
    
    if 'quizs' not in st.session_state or not st.session_state.quizs:
        st.warning("퀴즈가 없습니다. 먼저 퀴즈를 풀어주세요.")
        return
    
    for j, question in enumerate(st.session_state.quizs):
        res = json.loads(question["answer"])
        st.header(f"문제 {j+1}")
        st.write(f"**{res['quiz']}**")
        st.write(f"정답: {res['correct_answer']}")
        
        explanation = get_explanation(res['quiz'], res['correct_answer'])
        st.write(f"해설: {explanation}")
        st.markdown("---")
    
    if st.button('퀴즈 풀이 페이지로 돌아가기'):
        st.switch_page("pages/quiz_solve_page.py")

if __name__ == "__main__":
    quiz_review_page()

