from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME="gpt-4"


def main():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an expert translator from English to Swedish."), ("user", "{input}")]
    )
    llm = ChatOpenAI(model_name=MODEL_NAME)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    print("Input something in English to translate into Swedish:")
    text_to_translate = input()

    response = chain.invoke({"input": text_to_translate})

    print(response)


if __name__ == "__main__":
    main()
