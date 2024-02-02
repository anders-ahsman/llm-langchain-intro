from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
TEXT_FILE = "state_of_the_union.txt"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Use contents of text file as context
    with open(TEXT_FILE, "r") as f:
        context = f.read()

    setup = RunnableParallel(
        context=lambda _: context,  # context too long for model!
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
