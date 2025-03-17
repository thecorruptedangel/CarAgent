from flask import Flask, request, Response, session
from twilio.twiml.voice_response import VoiceResponse
import os
import datetime
from functools import lru_cache
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Load environment variables
load_dotenv()

# Global QA chain
qa_chain = None

# Load the document
def load_document(file_path):
    return TextLoader(file_path).load()

# Split document into chunks
def chunk_data(data, chunk_size=1000, chunk_overlap=20):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    ).split_documents(data)

# Define a cache for embedding texts
CACHE_SIZE = 10000

@lru_cache(maxsize=CACHE_SIZE)
def get_embeddings_cached(texts_tuple):
    texts = list(texts_tuple)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embedded_vectors = embeddings.embed_documents(texts)
    return embedded_vectors

# Convert embeddings to numpy array
def get_embeddings(texts):
    embeddings = get_embeddings_cached(tuple(texts))
    return np.array(embeddings, dtype=np.float32)

# Create and persist embeddings
def create_embeddings(chunks, persist_directory):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

# Setup QA chain
def setup_qa_chain(vector_store):
    system_message = (
        "You are an empathetic and helpful customer support agent. You are allowed to welcome users and be polite when user thanks or acknowledge or ending the call. You can also try to calm user down if they get frustrated. Behave like a personal assistant who talks like a human but use the technical knowledge from the available context while responding to user. User may say phrases such as 'thats all' or similar ones, in such cases user is actually happy with the resolution and is preparing to end the conversation. If the information is not in the context or previous queries, only respond '[UNKNOWN_QUERY]'. Keep responses short and in a single paragraph without special formatting."
    )

    prompt_template = PromptTemplate(
        template=f"{system_message}\n\nContext: {{context}}\n\nChat History: {{chat_history}}\n\nHuman: {{question}}\n\nAssistant:",
        input_variables=["context", "chat_history", "question"]
    )

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

# Bot function to handle user input and maintain chat history
def bot(query):
    chat_history = session.get('chat_history', [])

    # Process user query
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    answer = result['answer']

    # Handle unknown queries
    if '[UNKNOWN_QUERY]' in answer:
        answer = "Our team will get back to you shortly regarding this query."
        folder_path = 'unknown_queries'
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, 'queries.txt')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(file_path, 'a') as file:
            file.write(f"{timestamp}: {query}\n")

    # Update chat history
    chat_history.append((query, answer))
    session['chat_history'] = chat_history

    return answer

# Initialize embeddings and QA chain
def initialize_bot(file):
    global qa_chain
    try:
        document = load_document(file)
        chunks = chunk_data(document)
        vector_store = create_embeddings(chunks, './chroma_db')
        qa_chain = setup_qa_chain(vector_store)
        print("Bot initialized successfully")
    except Exception as e:
        print(f"Error initializing bot: {str(e)}")
        raise

@app.route("/voice", methods=['POST'])
def voice():
    try:
        response = VoiceResponse()
        response.say("Hello!, Welcome to Bosch. How can I help you today?", voice='alice')
        response.gather(input='speech', timeout=5, action='/gather')
        return Response(str(response), mimetype='text/xml')
    except Exception as e:
        print(f"Error in voice route: {str(e)}")
        return Response("An error occurred", status=500)

@app.route("/gather", methods=['POST'])
def gather():
    try:
        response = VoiceResponse()
        speech_result = request.form.get('SpeechResult', '')
        if speech_result:
            bot_response = bot(speech_result)
            response.say(bot_response, voice='alice')
            response.gather(input='speech', timeout=5, action='/gather')
        else:
            response.say("I didn't catch that. Could you please repeat?", voice='alice')
            response.gather(input='speech', timeout=5, action='/gather')
        return Response(str(response), mimetype='text/xml')
    except Exception as e:
        print(f"Error in gather route: {str(e)}")
        return Response("An error occurred", status=500)

@app.route("/status", methods=['POST'])
def status():
    try:
        call_status = request.form.get('CallStatus')
        call_sid = request.form.get('CallSid')
        print(f"Call SID: {call_sid}, Status: {call_status}")
        handle_call_status(call_status, call_sid)
        return Response("Status received", status=200)
    except Exception as e:
        print(f"Error in status route: {str(e)}")
        return Response("An error occurred", status=500)

def handle_call_status(status, sid):
    try:
        if status == "completed":
            print(f"Call {sid} has been completed.")
            # Clear the chat history for this session
            session.pop('chat_history', None)
        elif status == "failed":
            print(f"Call {sid} has failed.")
            # Clear the chat history for this session
            session.pop('chat_history', None)
        else:
            print(f"Call {sid} status: {status}")
    except Exception as e:
        print(f"Error in handle_call_status: {str(e)}")

if __name__ == "__main__":
    file_name = 'bosch.txt'
    try:
        initialize_bot(file_name)
        print("Documents Loaded")
        app.run(host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Failed to start the application: {str(e)}")