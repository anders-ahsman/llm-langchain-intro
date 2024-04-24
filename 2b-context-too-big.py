from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
TEXT_FILE = "king_lear.txt"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    # Create a prompt that includes the context and the question
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Use contents of text file as context
    with open(TEXT_FILE, "r") as f:
        context = f.read()

    context_and_question = RunnableParallel(
        context=lambda _: context,  # Context too big for model, will throw an error!
        question=RunnablePassthrough(),
    )

    # Combine into chain
    chain = context_and_question | prompt | llm

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = chain.invoke(user_input)
        print(response.content)


if __name__ == "__main__":
    main()
