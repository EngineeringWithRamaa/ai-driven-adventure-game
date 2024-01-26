Following is the workflow of AI-DRIVEN-ADVENTURE-GAME

Step 1 - DataStax Astra DB Integration:

Established connection with DataStax Astra DB using Python Cassandra driver.
Configured secure connection using the provided secure bundle path and application token.

Step 2 - VectorDB and Chat History:

Stored and managed conversation data as Vector Embeddings in VectorDB, with context expiration after 1 hour.
Implemented a chat message buffer/memory area using wrapper around Cassandra VectorDB.

Step 3 - LangChain Memory and LLMChain Integration:

Created a ConversationBufferMemory to act as a memory buffer for OpenAI model and LangChain.
Integrated LangChain with the Large Language Model (LLM) and the VectorDB.
This technique is what is called Retrieval Augmented Generation where LLM models are prompted with User Inputs and Actual facts
so that LLM model can generate more effective and reliable responses with Live Factual data.

Step 4 - Prompts for Adventure game

Developed a dynamic and engaging prompt for an adventure game scenario in the Whispering Woods.
Included rules for the game, such as choosing weapons, defining success paths, and handling death scenarios.

Step 5 - Gaming Loop:

Created an interactive loop allowing the AI to take turns and prompt the user for input in a dynamic adventure game.
Effectively utilized LangChain to generate AI responses based on user input.
