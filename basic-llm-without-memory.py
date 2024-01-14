from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "gpt-4"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = llm.invoke(user_input)
        print(response.content)


if __name__ == "__main__":
    main()
