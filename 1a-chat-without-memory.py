from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    # The context is basically the single input to the LLM.
    # Different models have different maximum context sizes.

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        # Only the last user input is used as context, so the LLM does not know about previous inputs.
        response = llm.invoke(user_input)
        print(response.content)


if __name__ == "__main__":
    main()
