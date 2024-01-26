from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate

import json
import os

def initialize_vector_db():
    try:

        # Establishing connection with VectorDB provided by DATASTAX
        session = Cluster(
            cloud={"secure_connect_bundle": os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"]},
            auth_provider=PlainTextAuthProvider("token", os.environ["ASTRA_DB_APPLICATION_TOKEN"]),
        ).connect()

        ASTRA_DB_KEYSPACE = "choose_your_own_adventure_game"

        # CassandraChatMessageHistory is a wrapper around VectorDB
        # Conversation or game narrations are stored as Vector Embeddings in VectorDB
        # Context is expired after 1 hr
        message_history = CassandraChatMessageHistory(
            session_id="anything",
            session=session,
            keyspace=ASTRA_DB_KEYSPACE,
            ttl_seconds=3600
        )

        message_history.clear()

        # Return both the session and message_history instances
        return session, message_history

    except Exception as e:
        print(f"Error connecting to VectorDB: {str(e)}")
        return None, None

def create_cass_buff_memory(vector_db_instance):
    if vector_db_instance is not None:
        # ConversationBufferMemory creates a buffer or memory for LangChain
        # LangChain injects this context-aware enhanced facts as inputs to the LLM model
        cass_buff_memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=vector_db_instance
        )
        return cass_buff_memory
    else:
        print("Error: VectorDB initialization failed.")
        return None

### Superb important step of the Adventure game ###
# Creating a Prompt to provide it to LLM model to ask it acts as a Game coach 
# Giving various criterias to make the game interesting & fun
# Giving LangChain memory also as input
# RAG - Retrieval Augmented Generation technique
# Very crucial step, this way of providing dynamic set of inputs to
# Pre-trained LLM models for improving accuracy of generated response
def create_prompt_template():
    template = """
    You are now the guide of a mystical journey in the Whispering Woods. 
    A traveler named Elara seeks the lost Gem of Serenity. 
    You must navigate her through challenges, choices, and consequences, 
    dynamically adapting the tale based on the traveler's decisions. 
    Your goal is to create a branching narrative experience where each choice 
    leads to a new path, ultimately determining Elara's fate. 

    Here are some rules to follow:
    1. Start by asking the player to choose some kind of weapons that will be used later in the game
    2. Have a few paths that lead to success
    3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game

    Here is the chat history, use this to understand what to say next: {chat_history}
    Human: {human_input}
    AI:"""

    # A PromptTemplate for interacting with OpenAI model
    # Includes both previous chat history and current user input to the OpenAI model
    # To enhance context awareness of the AI model to generate a more relevant & reliable response
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    return prompt

def initialize_language_model_chain():
    # Define OpenAI API key
    OPENAI_API_KEY = os.environ["OPENAI_API"]

    # Initialize OpenAI LLM (Large Language Model)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    # Initialize LLMChain with details of LLM model, Prompt, Memory in VectorDB
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=cass_buff_memory
    )

    return llm_chain

# Call the function to initialize VectorDB & create memory
session_instance, vector_db_instance = initialize_vector_db()

cass_buff_memory = create_cass_buff_memory(vector_db_instance)

prompt = create_prompt_template()

llm_chain = initialize_language_model_chain()

choice = "start"    

while True:
    # AI's turn
    response = llm_chain.predict(human_input=choice)
    print(response.strip())

    if "The End." in response:
        break

    # User's turn
    choice = input("Your reply: ")

# End of the game
print("Game Over.")