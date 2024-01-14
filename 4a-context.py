from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    prompt_with_context = """
    Answer the question based only on the following context:
    USA first visited the moon in 1950.

    Question:
    """

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = llm.invoke(prompt_with_context + user_input)
        print(response.content)


if __name__ == "__main__":
    main()
