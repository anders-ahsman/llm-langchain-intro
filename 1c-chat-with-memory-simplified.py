from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"


def main():
    llm = ChatOpenAI(model_name=MODEL_NAME)
    conversation_chain = ConversationChain(llm=llm)

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            break

        response = conversation_chain.predict(input=user_input)
        print(response)


if __name__ == "__main__":
    main()
