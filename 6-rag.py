from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL_NAME = "gpt-3.5-turbo"
TEXT_FILE = "state_of_the_union.txt"


def main():
    loader = TextLoader(TEXT_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)

    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model_name=MODEL_NAME)

    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    setup = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    )
    chain = setup | prompt | llm

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = chain.invoke(user_input)
        print(response.content)


if __name__ == "__main__":
    main()
