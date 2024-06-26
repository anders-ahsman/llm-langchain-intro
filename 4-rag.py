from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 150
SEARCH_NUM_RESULTS = 3
TEXT_FILE = "state_of_the_union.txt"


def main():
    documents = load_documents()
    retriever = get_vectordb_retriever(documents)
    chain = setup_chain(retriever)

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        call_chain_and_print_results(chain, user_input)


def load_documents() -> list[Document]:
    """
    Load documents from text file and split them into chunks.
    """

    loader = TextLoader(TEXT_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents_split = text_splitter.split_documents(documents)

    return documents_split


def get_vectordb_retriever(documents: list[Document]) -> VectorStoreRetriever:
    """
    Create a vector database from the documents and return a retriever to the database.
    """

    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents, embeddings_model)

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": SEARCH_NUM_RESULTS,
        },
    )

    return retriever


def setup_chain(retriever: VectorStoreRetriever) -> RunnableSequence:
    """
    Setup the chain that will be used to answer questions.
    """

    llm = ChatOpenAI()

    # Create a prompt that includes the context and the question.
    template = """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Combine into RAG chain:
    # 1. Retrieve documents relevant to the question (Retrieval)
    # 2. Generate response to question using retrieved documents as context (Augmented Generation)
    chain: RunnableSequence = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def call_chain_and_print_results(chain: RunnableSequence, user_input: str) -> None:
    """
    Call the chain with the user input and print the results (streaming).
    """

    for s in chain.stream(user_input):
        print(s, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
