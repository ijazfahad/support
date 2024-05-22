# app.py
import os
import openai
from flask import Flask, request, jsonify, render_template
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
import constants
import logging

app = Flask(__name__)

# Set the logging level to DEBUG to capture detailed information
app.logger.setLevel(logging.DEBUG)

# Add a StreamHandler to log messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
app.logger.addHandler(stream_handler)


os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False

if PERSIST and os.path.exists("persist"):
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
#     loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": "persist"},
            embedding=OpenAIEmbeddings()
        ).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(
            embedding=OpenAIEmbeddings()
        ).from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

# Define a route to handle GET requests
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define a route to handle POST requests
@app.route('/', methods=['POST'])
def handle_post():
    try:
        data = request.json
        if data and 'query' in data:
            query = data['query']
            app.logger.debug('Received query: %s', query)

            # Process the query here (replace this with your logic)
#             response = {'message': 'Received query', 'query': query}
            if query in ['quit', 'q', 'exit']:
                return jsonify({'answer': 'Goodbye!'}), 200

            result = chain({"question": query, "chat_history": chat_history})
            chat_history.append((query, result['answer']))
            return jsonify({'answer': result['answer']})

#             return jsonify(response), 200
        else:
            app.logger.error('Invalid or missing query data')
            return jsonify({'error': 'Invalid or missing query data'}), 400
    except Exception as e:
        app.logger.exception('An error occurred while processing the request: %s', str(e))
        return jsonify({'error': 'An internal server error occurred'}), 500

# Define an error handler for all exceptions
@app.errorhandler(Exception)
def handle_error(e):
    app.logger.error('An error occurred: %s', str(e))
    return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
