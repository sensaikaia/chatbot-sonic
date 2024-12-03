# import os
# from dotenv import find_dotenv, load_dotenv
# import openai

# from langchain_community.vectorstores import FAISS

# from langchain_openai import ChatOpenAI

# from langchain.prompts import PromptTemplate
# from langchain_openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA

# # Load environment variables
# dotenv_path = find_dotenv()
# load_dotenv(dotenv_path)
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Set OpenAI API key
# openai.api_key = openai_api_key

# # Load FAISS vector store
# def load_vector_store():
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     return vector_store

# vector_store = load_vector_store()

# # Initialize OpenAI chat model (gpt-3.5-turbo)
# sonic_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=800)


# prompt_template = """
# You are SONiC Scout, a helpful and friendly guide for SONiC Network Engineers and newcomers. Explain answers clearly and supportively based on the provided context. If you do not know the answer, say, "I'm still learning, but I’m here to help as best I can!"

# Context:
# {context}

# Question: {question}

# Answer:
# """

# # Setup a PromptTemplate and RetrievalQA chain
# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Use RetrievalQA for question-answering with context retrieval from FAISS
# qa_chain = RetrievalQA.from_chain_type(
#     llm=sonic_model,
#     chain_type="stuff",  # Choose 'map_reduce' or 'refine' if more complex processing is needed
#     retriever=vector_store.as_retriever(),
#     chain_type_kwargs={"prompt": prompt}
# )

# # Conversational loop with history tracking for context
# def sonic_chatbot():
#     conversation_history = []

#     print("SONiC Scout: I have been developed by xFlow SONiC Engineers. Let's talk SONiC!")

#     while True:
#         print("-" * 50)  # Divider line for better readability
#         user_input = input("You: ")

#         # Exit condition for the chat
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("SONiC Scout: Though I am learning, I hope I answered your questions. See you! Goodbye!")
#             break

#         # Generate embedding for the user input
#         user_input_embedding = vector_store.embeddings.embed_query(user_input)

#         # Search for the most relevant documents based on the user query
#         docs = vector_store.similarity_search_by_vector(user_input_embedding)

#         # If no relevant context is found
#         if not docs:
#             print("SONiC Scout: I am still learning, and this answer is not in my current knowledge base.\n")
#             continue

#         # Extract content from retrieved documents
#         context = "\n".join([doc.page_content for doc in docs])

#         # Maintain conversational history by incorporating prior exchanges in context
#         full_context = "\n".join([f"User: {entry['user']}\nSONiC: {entry['assistant']}" for entry in conversation_history])
#         full_context += f"\nContext:\n{context}"

#         # Run the QA chain with combined context and user question
#         #response = qa_chain({"context": full_context, "question": user_input})
#         response = qa_chain.invoke({"query": user_input, "context": full_context})

#         # Get the model's answer
#         model_response = response.get("result", "I couldn't find an answer.")

#         # Output the answer in a chat format
#         print(f"SONiC: {model_response}\n")

#         # Append this conversation to the history
#         conversation_history.append({"user": user_input, "assistant": model_response})

# if __name__ == "__main__":
#     sonic_chatbot()








import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load FAISS vector store
def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_vector_store()

# Initialize ChatGPT model
sonic_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

# Prompt template with chain-of-thought prompting
prompt_template = """
You are SONiC Scout, a helpful and friendly guide for SONiC Network Engineers and newcomers. 
Explain answers clearly and supportively based on the provided context.

Context:
{context}

<user_query> {question} </user_query>

<chain_of_thought>
Think through the problem logically, breaking it down step by step.
</chain_of_thought>

<answer>
Provide a well-structured answer based on the reasoning above.
</answer>
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Use RetrievalQA for question-answering with context retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=sonic_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Conversational loop for SONiC Scout chatbot
def sonic_chatbot():
    conversation_history = []
    print("SONiC Scout: I have been developed by xFlow SONiC Engineers. Let's talk SONiC!")

    while True:
        print("-" * 50)
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("SONiC Scout: See you! Goodbye!")
            break

        # Generate embedding and retrieve relevant documents
        user_input_embedding = vector_store.embeddings.embed_query(user_input)
        docs = vector_store.similarity_search_by_vector(user_input_embedding, k=8)
        
        # Check for retrieved documents
        if not docs:
            print("SONiC Scout: I am still learning, and this answer is not in my current knowledge base.\n")
            continue

        # Limit context to a manageable size
        context = "\n".join([doc.page_content for doc in docs])[:2000]

        # Maintain conversational history
        full_context = "\n".join(
            [f"User: {entry['user']}\nSONiC: {entry['assistant']}" for entry in conversation_history]
        ) + f"\nContext:\n{context}"

        # Run the QA chain
        response = qa_chain.invoke({"query": user_input, "context": full_context})
        model_response = response.get("result", "I couldn't find an answer.")

        # Display the response
        print(f"SONiC: {model_response.strip()}\n")

        # Store in conversation history
        conversation_history.append({"user": user_input, "assistant": model_response})

if __name__ == "__main__":
    sonic_chatbot()
