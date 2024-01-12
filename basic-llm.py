from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME="gpt-4"


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an helpful chatbot that likes to tell jokes."),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model_name=MODEL_NAME)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    print("Ask a question:")
    user_input = input()

    response = chain.invoke({"input": user_input})
    print(response)


if __name__ == "__main__":
    main()
