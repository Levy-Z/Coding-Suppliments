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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

VECTORSTORE_LOAD_PATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books/db_store"

SUPPORTED_REASONING_MODES = {
    "direct",
    "concise_rationale",
    "cot_verbose"
}

# Getting a path to our trading books
os.chdir(os.getcwd())
books_dir = "trading_books"
list_of_books = os.listdir(books_dir)


def read_trading_books(books_path, num_books=None):
    """
    Load PDF books from the specified directory and return page contents.
    """
    print("READING TRADING BOOKS")
    extracted_data = []

    os.chdir(books_path)
    books_to_read = os.listdir(os.getcwd())[:num_books] if num_books else os.listdir(os.getcwd())

    for book in books_to_read:
        print(f"Loading: {book}")
        loader = PyPDFLoader(book)
        data = loader.load()
        extracted_data.extend(data)

    return [doc.page_content for doc in extracted_data]


def split_trading_books(book_texts, splitter=None):
    """
    Split raw book text into smaller chunks for embedding/retrieval.
    """
    print("SPLITTING TRADING BOOKS")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    ) if splitter is None else splitter

    chunks = []
    for text in book_texts:
        chunk = text_splitter.split_text(text)
        chunks.extend(chunk)

    return chunks


def get_embeddings(chunked_docs=None, embedder=None, use_gpu=False, only_return_embedder=False):
    """
    Return the embedding model, and optionally embeddings for chunked docs.
    """
    embeddings_model = (
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        if embedder is None else embedder
    )

    if only_return_embedder:
        print("GETTING EMBEDDER")
        return embeddings_model

    print("GETTING EMBEDDER AND EMBEDDINGS")
    embeddings = embeddings_model.embed_documents(chunked_docs)
    return embeddings_model, np.array(embeddings)


def get_vectorstore(chunked_data=None, save_path=None, load_path=None, embedder=None):
    """
    Either load an existing FAISS vectorstore or create a new one.
    """
    if load_path:
        print("GETTING PRELOADED VECTORSTORE")
        db_store = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
        return db_store

    print("CREATING VECTORSTORE")
    embeddings_model, embeddings = get_embeddings(chunked_data, embedder)
    embeddings = np.array(embeddings)

    dim = embeddings.shape[1]
    db_index = faiss.IndexFlatL2(dim)
    db_index.add(embeddings)

    docs = [Document(page_content=text) for text in chunked_data]
    docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=db_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    if save_path:
        vectorstore.save_local(save_path)

    return vectorstore


def get_retreiver(db_store, k=5):
    """
    Return retriever from FAISS vectorstore.
    """
    return db_store.as_retriever(
        search_kwargs={"k": k}
    )


def wrap_prompt_with_reasoning(system_prompt: str, mode: str = "concise_rationale") -> str:
    """
    Add structured reasoning instructions to the system prompt.

    Modes:
        - direct: concise answer only
        - concise_rationale: short explanation + final answer + sources
        - cot_verbose: more detailed step-by-step reasoning
    """
    if mode not in SUPPORTED_REASONING_MODES:
        raise ValueError(
            f"Unsupported reasoning mode '{mode}'. "
            f"Choose from: {SUPPORTED_REASONING_MODES}"
        )

    common_rules = """
Use the retrieved context as your primary source of evidence.
If the context is insufficient, you may use general trading knowledge, but clearly label it as general knowledge.
Do not fabricate citations or claim the context supports something it does not.
If the answer is uncertain, say so explicitly.
"""

    if mode == "direct":
        return system_prompt + common_rules + """

Respond using the following format:

FINAL ANSWER:
Provide a direct and concise answer.

SOURCES:
- Context: identify what came from the retrieved context
- General knowledge: clearly label any outside knowledge
"""

    if mode == "cot_verbose":
        return system_prompt + common_rules + """

Use a structured reasoning process before producing the answer.

THOUGHT PROCESS:
1. Understand the user's question.
2. Identify the most relevant retrieved context.
3. Connect the retrieved evidence to the question.
4. Apply trading or quantitative finance knowledge only when necessary.
5. Double-check assumptions, logic, or calculations.
6. Synthesize the conclusion.

FINAL ANSWER:
Provide a concise final answer.

SOURCES:
- Context: identify supporting evidence from retrieved context
- General knowledge: clearly label any outside knowledge
"""

    return system_prompt + common_rules + """

Respond using the following format:

RATIONALE:
Briefly explain the main reasoning steps in 3-6 sentences. Keep it concise and grounded in the retrieved context.

FINAL ANSWER:
Provide a clear and direct answer.

SOURCES:
- Context: identify what came from the retrieved context
- General knowledge: clearly label any outside knowledge
"""


def create_rag_chain(retriever, prompt, llm=None, llm_kwargs=None):
    """
    Create the RAG chain that retrieves relevant documents and generates an answer.
    """
    llm_kwargs = llm_kwargs or {
        "model": "gemini-2.0-flash",
        "temperature": 0.5
    }

    model = llm(**llm_kwargs) if llm else ChatGoogleGenerativeAI(**llm_kwargs)

    system_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(model, system_prompt)

    rag_chain = (
        RunnableMap({
            "input": lambda x: x["input"],
            "context": lambda x: retriever.invoke(x["input"])
        })
        | RunnableLambda(lambda x: {
            "answer": chain.invoke({
                "input": x["input"],
                "context": x["context"]
            }),
            "retrieved_docs": x["context"]
        })
    )

    return rag_chain


def create_whole_pipeline(
    system_prompt,
    vectorstore_load_path=None,
    documents_dir=None,
    vectorstore_save_path=None,
    reasoning_mode="concise_rationale",
    retriever_k=5,
    llm=None,
    llm_kwargs=None
):
    """
    Create the full RAG pipeline with optional reasoning mode selection.
    """
    embedder = get_embeddings(only_return_embedder=True)

    if vectorstore_load_path:
        vectorstore = get_vectorstore(load_path=vectorstore_load_path, embedder=embedder)
    else:
        data = read_trading_books(documents_dir)
        chunked = split_trading_books(data)
        vectorstore = get_vectorstore(
            chunked_data=chunked,
            save_path=vectorstore_save_path,
            embedder=embedder
        )

    retreiver = get_retreiver(vectorstore, k=retriever_k)

    prompt_with_context = system_prompt + "\n\nRetrieved Context:\n{context}"
    wrapped_prompt = wrap_prompt_with_reasoning(
        prompt_with_context,
        mode=reasoning_mode
    )

    rag_chain = create_rag_chain(
        retreiver,
        wrapped_prompt,
        llm=llm,
        llm_kwargs=llm_kwargs
    )

    return rag_chain


if __name__ == "__main__":
    base_system_prompt = """
You are an expert in trading strategies and quantitative finance.
Your role is to guide and assist traders by answering questions, evaluating strategies,
and explaining whether a strategy appears profitable, reasonable, or risky.

Prioritize the retrieved context when answering.
When relevant, distinguish clearly between:
1. information supported by retrieved context
2. general knowledge from trading and investing
"""

    reasoning_mode = "concise_rationale"  # Options: direct, concise_rationale, cot_verbose

    rag_chain = create_whole_pipeline(
        system_prompt=base_system_prompt,
        vectorstore_load_path=VECTORSTORE_LOAD_PATH,
        reasoning_mode=reasoning_mode,
        retriever_k=5
    )

    sample_prompt = (
        "Give me some sample trading strategies, and tell me whether that information "
        "comes from the retrieved context or your own general knowledge."
    )

    ai_response = rag_chain.invoke({
        "input": sample_prompt
    })

    print("===== MODEL ANSWER =====")
    print(ai_response["answer"])

    print("\n===== RETRIEVED DOCUMENTS =====")
    for i, doc in enumerate(ai_response["retrieved_docs"], start=1):
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"\nDocument {i}: {preview}...")
