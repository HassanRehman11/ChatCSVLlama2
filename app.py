import streamlit as st
from utils import load_llm, load_vector
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

st.title("Employee Book Bot based on CSV")
st.markdown("<h3 style='text-align: center; color: black;'>Built by <a href='https://github.com/hassanrehman11'>Hassan Rehman </a></h3>", unsafe_allow_html=True)

db = load_vector()
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello I am an Employee Bot. You can ask anything related to resources!"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hello!!!"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Ask any thing about Pokemon", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
