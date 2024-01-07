import os

import openai
import pinecone
import streamlit as st
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.pinecone import Pinecone

from custom_callback import WriteCallBackManager


template = """You are an AI assistant for answering questions about the Consitution of nepal.
You are given the following extracted parts of a long document and a question. Provide answer in a simple terms, try to explain each difficult terms in as simple as possible.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the Consitution of nepal, politely inform them that you are tuned to only answer questions about Consitution of nepal.
Lastly, answer the question as if you were expert of nepal consitution.
Question: {question}
=========
{context}
=========
Answer in text:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])

# Define a function to get the page number of a document
def get_page_number(document):
    return document.metadata["page_number"]


def load_retriever():
    # question= "How many years of imprisionment will be for rape?"
    # with open("vectorstore.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)
    # docs = vectorstore.similarity_search(question,include_metadata=True)
    # print(docs)
    # print(get_page_number(docs[0]))
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"],environment="gcp-starter")
    index_name=pinecone.Index(index_name="kanun")
    embedding = OpenAIEmbeddings()
    data = Pinecone(index_name,embedding=embedding.embed_query,text_key="text")
    # result = data.similarity_search(query=question)
    # print(result)
    return data
    # retriever = VectorStoreRetriever(vectorstore=vectorstore)
    # return retriever

def get_custom_prompt_qa_chain():
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    question_llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",streaming=True,callbacks=[WriteCallBackManager(st.container())])
    retriever = load_retriever()
    question_generator = LLMChain(
        llm=question_llm, prompt=CONDENSE_QUESTION_PROMPT
    )
    # print(QA_PROMPT)
    doc_chain = load_qa_chain(
        llm, chain_type="stuff", prompt=QA_PROMPT
    )

    qa = ConversationalRetrievalChain(
        # vectorstore=vectorstore,
        retriever=retriever.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        return_source_documents=True,
    )
    return qa

def main():
    # show user input
    st.set_page_config(page_title="Wakil-G",page_icon="🙃",menu_items={
        'About': "This is an AI application that responds based on the document provided. Anything said by the application should not be cited directly as a source, as it is still under development and may contain errors."
    })
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "disable_input" not in st.session_state:
        st.session_state.disable_input = False
    
    for histories in st.session_state.chat_history:
        for i in range(0,len(histories)):
            if (i == 0):
                with st.chat_message("user"):
                    st.markdown(histories[i])
            else:
                with st.chat_message("assistant"):
                    st.markdown(histories[i])
    user_question = st.chat_input("What are my basic rights?",disabled=st.session_state.disable_input)
    if user_question:
        try:
            st.session_state.disable_input = True
            with st.chat_message("user"):
                st.markdown(user_question)
            chain = get_custom_prompt_qa_chain()
            response =  chain({"question":user_question,"chat_history":st.session_state.chat_history})
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            st.session_state.disable_input = False
            st.session_state.chat_history.append((user_question, response['answer']))
        except:
            st.write("Something went wrong, Please report <a href='https://github.com/prakash-O4/wakil-g/issues' target='_blank'>here</a>")
        # st.write(st.session_state.chat_history)
        # print(response['source_documents'])
        # st.write(response['answer'])
    else:
        # Create an empty element to center content
        start_conversation_placeholder = st.empty()
        
        # Define a meaningful message for an empty chat
        empty_chat_message = "Let's start a conversation. Ask me about Constitution and Criminal Law of Nepal"

        # Use markdown with CSS to style the message and add an icon
        start_conversation_placeholder.markdown(
            f"""
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh;">
            <div style="text-align: center;">
                <h2 style="font-size: 34px;"> 🙃 Wakil-G</h2>
                <h2 style="font-size: 18px;"> Pocket-Size Nepal Law Guru! </h2>
                <h2 style="font-size: 20px;">{empty_chat_message}</h2>
            </div>
            <div style="border: 1px solid #ccc; padding: 10px; margin: 5px; width: 100%;">
                <p style="font-weight: bold;">Demo questions:</p>
                <p>What are my basic rights?</p>
                <p>What is the process of filing a FIR?</p>
                <p>Can you list me the punishable offences?</p>
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
        """
        <div style="font-size: 12px; margin-top: 20px; color: #777;">
            Disclaimer: This is an AI application that responds based on <a href='https://www.moljpa.gov.np/en/wp-content/uploads/2018/12/Penal-Code-English-Revised-1.pdf' target='_blank'>provided documents</a>.. 
            The information provided may not have a direct source reference, and it is under maintenance. 
            It may contain bugs or inaccuracies, if found so please report <a href='https://github.com/prakash-O4/wakil-g/issues' target='_blank'>here</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )
if __name__ == '__main__':
    main()




