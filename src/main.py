from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
#Streamlit Setup:
#The streamlit library is used to create a user interface for the chatbot application.
#A subheader is displayed to indicate that it's a chatbot using Langchain, ChatGPT, Pinecone, and Streamlit.
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")
#Session State Management:
#Session state variables 'responses' and 'requests' are initialized using Streamlit's st.session_state. 
#These variables are used to store the chat history.
#if 'responses' not in st.session_state: checks if the session state variable 'responses' has been initialized.
#If 'responses' is not found in the session state, it means it hasn't been initialized yet.
#In that case, st.session_state['responses'] = ["How can I assist you?"] initializes the 'responses' variable with a list containing a default response message, in this case, "How can I assist you?".
#This ensures that if the 'responses' variable hasn't been set before, it gets initialized with a default value.
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []
#The ChatOpenAI model from Langchain is initialized with the specified model name ("gpt-3.5-turbo") and OpenAI API key.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")
#Conversation Chain Initialization
#A conversation chain is set up using Langchain. It involves defining a conversation memory buffer (buffer_memory) to store 
#previous messages, setting up prompt templates for the chatbot, and initializing the conversation chain itself.
#This memory is likely used to store previous messages in the conversation for context or history tracking.
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

#The template string contains instructions for the user to answer questions truthfully using the provided context. 
#It also instructs the user to say "I don't know" if the answer is not contained within the provided text.
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

#The template string {input} indicates that the input from the user should be directly included in the prompt.
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
# This line creates a chat prompt template from a list of message templates.The list includes the system message template, 
#a placeholder for message history (MessagesPlaceholder), and the human message template.
#The MessagesPlaceholder template is used to include the history of previous messages in the conversation.
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
#ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True): This line initializes a conversation chain.
#The conversation chain is initialized with the following parameters:
#memory=st.session_state.buffer_memory: The conversation memory buffer is set to the buffer memory initialized earlier in the session state.
#prompt=prompt_template: The prompt template for the conversation is set to the prompt template created earlier.
#@llm=llm: The chatbot model (llm) initialized earlier is used for generating responses.
#verbose=True: This parameter enables verbose logging, which may provide additional information during the conversation.

# this block of code sets up the templates for system messages and human messages, creates a prompt template for the conversation,
# and initializes the conversation chain with the specified parameters.


# container for chat history
response_container = st.container() 
# container for text box. creates another Streamlit container (textcontainer) to hold the text input field for user queries
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):#When a user enters a query (if query:),It displays a spinner ("typing...") to indicate that the chatbot is processing the query.
            conversation_string = get_conversation_string()#It retrieves the conversation history as a string (conversation_string)
            #using the get_conversation_string() function.
            # st.code(conversation_string)
            #The query is refined using the query_refiner() function based on the conversation history.
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")#The refined query is displayed as a subheader ("Refined Query:") using st.subheader() and st.write().
            st.write(refined_query)
            #The context for the query is retrieved using the find_match() function.
            context = find_match(refined_query)
            # print(context)
            #The conversation chain predicts a response (response) based on the context and the user's query
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            ##The user query and bot response are appended to the session state variables requests and responses, respectively
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
        #Inside the response_container, the code iterates over the stored responses (st.session_state['responses']) and displays
        # each response using the message() function. The key parameter is used to uniquely identify each message.
with response_container:
    if st.session_state['responses']:
        #If the index i is less than the length of the stored user queries (st.session_state['requests']), the corresponding user 
        #query is also displayed using the message() function with the is_user=True parameter.

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          #Overall, this block of code creates a user interface for the chatbot application, allowing users to input queries, 
          #view the refined query, see the bot's response, and review the chat history.