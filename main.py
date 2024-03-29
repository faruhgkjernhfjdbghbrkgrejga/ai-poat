import os
import streamlit as st
from langchain.chat_models import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(api_key=openai_api_key)

st.title('인공지능 시인')

content = st.text_input('주제')

if st.button('시 작성'):
    with st.spinner('시 쓰는 중'):
        result = chat_model.predict(content + "에 대한 시를 써줘")
        st.write(result)




