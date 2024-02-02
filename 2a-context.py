from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    # Create a prompt that includes the context and the question
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Setup the context and placeholder for the question.
    context_and_question = RunnableParallel(
        context=lambda _: "USA first visited the moon in 1950.",  # static context as example
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
