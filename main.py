
from constant import OPENAI_KEY

# load excel files
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
loader = UnstructuredCSVLoader(file_path='datas/RAG_data_test.csv')
docs = loader.load()

# Split text documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30,
)
documents = text_splitter.split_documents(docs)



# create embeddings
from langchain_openai import OpenAIEmbeddings
openai = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

from langchain_pinecone import PineconeVectorStore
from constant import PINECONE_INDEX_NAME
vectorstore = PineconeVectorStore.from_documents(documents, index_name=PINECONE_INDEX_NAME, embedding=openai)

# memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# set query
from langchain_openai import ChatOpenAI  

llm = ChatOpenAI(  
    openai_api_key=OPENAI_KEY,  
    model_name='gpt-3.5-turbo',  
    temperature=0.0  
) 

from langchain.chains import RetrievalQA  
qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=vectorstore.as_retriever()  
)

querying_after_response = True

while querying_after_response:
    # query as-self
    query = input("Vous : ")
    print('\n')

    # return response
    response = qa.invoke(query)

    print("""Chatbot: {}
    """.format(response["result"]))