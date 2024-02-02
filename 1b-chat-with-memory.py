from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"


def main():
    conversation = "System: You are an helpful chatbot that likes to tell jokes."
    llm = ChatOpenAI(model_name=MODEL_NAME)

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        # Add the user input to the conversation
        conversation += f"\nHuman: {user_input}"
        print(f"*** Conversation so far:\n{conversation}\n***\n")

        # Here the entire conversation is used as the context, not just the last user input.
        # That way the LLM can use the entire conversation to generate a response.
        response = llm.invoke(conversation)
        print(response.content)

        # Add the LLM's response to the conversation
        conversation += f"\n{response.content}"


if __name__ == "__main__":
    main()
