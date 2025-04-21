import os
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="📚 Lots of love for smriti, hehe", layout="centered")

from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain

# ✅ Load environment variables
load_dotenv()

# ✅ Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ Initialize LangChain components
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

qa_chain = load_chain()

# ✅ Streamlit UI setup
st.title("📚 Lots of love for smriti, hehe")

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chat input
user_input = st.chat_input("Ask something about Nationalism, Development, etc...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    result = qa_chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])

    # Save to history
    st.session_state.chat_history.append((user_input, result["answer"]))

# ✅ Optional: show conversation history
with st.expander("🕘 Chat History"):
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
