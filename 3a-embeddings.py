from langchain_openai.embeddings import OpenAIEmbeddings


def main():
    embeddings_model = OpenAIEmbeddings()

    text = "Hello, world!"
    query_result = embeddings_model.embed_query(text)

    print(f"Text: {text}")
    print(f"Embedding length (number of dimensions): {len(query_result)}")
    print(f"First 5 numbers of embedding: {query_result[:5]}")


if __name__ == "__main__":
    main()
