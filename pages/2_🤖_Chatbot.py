import streamlit as st

from chatbot_client import Client

st.title("Random Chatbot 🤖")
with st.expander("ℹ️ Disclaimer"):
    st.caption("Puedo robar tus datos si me das mucha información personal 😈")


client = Client()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
