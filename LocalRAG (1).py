import os
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

VECTORSTORE_LOAD_PATH = '/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books/db_store'

# Getting a path to our trading books
os.chdir(os.getcwd())
books_dir = "trading_books"
list_of_books = os.listdir(books_dir)

# Reading in our trading books using LangChain's PyPDF Loader
def read_trading_books(books_path, num_books = None):
    print("READING TRADING BOOKS")
    extracted_data = []

    # First, we want to switch to the directory where the books are
    os.chdir(books_path)

    books_to_read = os.listdir(os.getcwd())[:num_books] if num_books else os.listdir(os.getcwd())
    
    for book in books_to_read:
        print(book)
        loader = PyPDFLoader(book)
        data = loader.load()

        extracted_data.extend(data)


    return [i.page_content for i in extracted_data] # Return type is a list of lists where each inner list is a list of documents from that specific book

def split_trading_books(book_texts, splitter = None):
    print("SPLITTING TRADING BOOKS")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 750,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex=False
    ) if not splitter else splitter

    chunks = []

    for text in book_texts:
        chunk = text_splitter.split_text(text)
        chunks.extend(chunk)

    return chunks


def get_embeddings(chunked_docs = None, embedder = None, use_gpu = False, only_return_embedder = False):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
        ) if not embedder else embedder

    if not only_return_embedder:
        print("GETTING EMBEDDER AND EMBEDDINGS")
        embeddings = embeddings_model.embed_documents(chunked_docs)
        return embeddings_model, np.array(embeddings)
    else:
        print("GETTING EMBEDDER")
        return embeddings_model


def get_vectorstore(chunked_data: str = None, save_path: str = None, load_path : str = None, embedder = None): # Here we either create a new vectorsore, or load one in from memory
    if load_path:
        print("GETTING PRELOADED VECTORSTORE")
        db_store = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
        return db_store
    else:
        print("CREATING VECTORSTORE")
        embeddings_model, embeddings = get_embeddings(chunked_data, embedder)
        embeddings = np.array(embeddings)
        dim = embeddings.shape[1]
        db_index = faiss.IndexFlatL2(dim)

        gpu_index = db_index
        gpu_index.add(embeddings)

        docs = [Document(page_content=text) for text in chunked_data]
        docstore = InMemoryDocstore({
            str(i): docs[i] for i in range(len(docs))
        })
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}

        vectorstore = FAISS(
            embedding_function=embeddings_model,
            index=db_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        vectorstore.save_local(save_path)
        return vectorstore


def get_retreiver(db_store, k=5):
    return db_store.as_retriever(
        search_kwargs={
            "k": k,
        }
    )

def create_rag_chain(retriever, prompt, llm = None, llm_kwargs = None):
    model = llm(**llm_kwargs) if llm else ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(model, system_prompt)

    rag_chain = (
        RunnableMap({
            "input": lambda x: x["input"],
            "context": lambda x: retriever.get_relevant_documents(x["input"])
        })
        | RunnableLambda(lambda x: {
            "answer": chain.invoke({"input": x["input"], "context": x["context"]}),
            "retrieved_docs": x["context"]
        })
    )

    return rag_chain


def create_whole_pipeline(system_prompt, vectorstore_load_path = None, documents_dir = None, vectorstore_save_path = None):
    embedder = get_embeddings(only_return_embedder=True)
    if vectorstore_load_path:
        vectorstore = get_vectorstore(load_path=vectorstore_load_path, embedder=embedder)
    else:
        data = read_trading_books(documents_dir)
        chunked = split_trading_books(data)
        vectorstore = get_vectorstore(
            chunked_data=chunked,
            save_path=vectorstore_save_path
        )

    retreiver = get_retreiver(vectorstore)
    prompt = system_prompt + "\n\n{context}"
    rag_chain = create_rag_chain(retreiver, prompt)
    return rag_chain


if __name__ == "__main__":
    embedder = get_embeddings(only_return_embedder=True)

    vectorstore = get_vectorstore(load_path=VECTORSTORE_LOAD_PATH, embedder=embedder)
    retreiver = get_retreiver(vectorstore)

    prompt = """
    You are an expert on trading strategies and quantitative trading, and your job is to guide and assist traders, evaluating
    their strategies and determining, whether or not they are profitable, as well as answering their questions.

    Formulate your answers based on both the following context as well as your own knoweledge about trading and investments.

    \n\n{context}
    """

    rag_chain = create_rag_chain(retreiver, prompt)

    sample_prompt = "Give me some sample trading strategies, and tell me if you got that information from the context or your own knoweledge"

    ai_response = rag_chain.invoke({
        "input": sample_prompt
    })

    print(ai_response['answer'])
    
