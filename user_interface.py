import streamlit as st
from retrieval_system.retrieval_pipeline import pipeline
from llm.gpt_api import openai_gpt
import time
import os
import psutil

st.set_page_config(layout="wide")

if not pipeline.system_set:
    with st.spinner('Loading system...'):
        pipeline.setup_system('data/ds-medium-articles.csv')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
    exit_app = st.button("Shut Down")
    
    if exit_app:
        pipeline.loader.client.close()
        time.sleep(5)
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()

col1, col2 = st.columns(2, gap='medium')

with col1:
    st.title("💬 Chatbot")

returned_chunks = None

if prompt := st.chat_input():
    returned_chunks = pipeline.get_context(prompt)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    with col1:
        with st.container(height=650):
            st.session_state.messages.append({"role": "user", "content": prompt})
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            if not openai_api_key:
                st.info("Please add your OpenAI API key to chat.")
            else:
                response = openai_gpt(openai_api_key, [hit.payload['content'] for hit in returned_chunks], prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.chat_message("assistant").write(response)

with col2:
    st.title("📃 Retrieved chunks")
    if returned_chunks:
        with st.container(height=650):
            for hit in returned_chunks:
                    st.markdown(f"### Article title: {hit.payload['title']}")
                    st.markdown(f"**Score:** {round(hit.score, 2)}")
                    st.markdown(f"**Chunk content**: {hit.payload['content']}")