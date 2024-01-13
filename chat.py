from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME="gpt-4"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)
    conversation = ConversationChain(llm=llm)

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            break

        response = conversation.predict(input=user_input)
        print(response)


if __name__ == "__main__":
    main()
