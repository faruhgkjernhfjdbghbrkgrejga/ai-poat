# from dotenv import load_dotenv
# load_dotenv()

import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 설정해주세요.")

# ChatOpenAI 클래스의 인스턴스를 생성합니다.
chat_model = ChatOpenAI(api_key=openai_api_key)

st.title('인공지능 시인')

content = st.text_input('주제')

if st.button('시 작성'):
    with st.spinner('시 쓰는 중'):
        result = chat_model.predict(content + "에 대한 시를 써줘")
        st.write(result)




