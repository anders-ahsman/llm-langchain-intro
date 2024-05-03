from langchain_community.vectorstores import LanceDB
from langchain_openai.embeddings import OpenAIEmbeddings

import lancedb
from common import EXAMPLE_TEXTS

SEARCH_NUM_RESULTS = 3


def main():
    embeddings_model = OpenAIEmbeddings()

    table, vectorstore = get_table_and_vectorstore(embeddings_model)

    # Add example texts to the database
    vectorstore.add_texts(EXAMPLE_TEXTS)

    # Display the contents of the database
    print("Database contents:")
    print(table.to_pandas(), "\n")

    while True:
        query = input("Enter a search query (or type 'exit'): ")
        if query == "exit":
            return

        # Perform a similarity search and print the results
        result = vectorstore.similarity_search(query, k=SEARCH_NUM_RESULTS)
        for r in result:
            print(r.page_content)
        print()


def get_table_and_vectorstore(embeddings_model):
    # Create a LanceDB table, overwrite if it already exists.
    db = lancedb.connect("./lancedb")
    table = db.create_table(
        "my_table",
        data=[
            {
                "vector": embeddings_model.embed_query("hello world"),
                "text": "hello world",
                "id": "1",
            }
        ],
        mode="overwrite",
    )
    vectorstore = LanceDB(table, embeddings_model)

    return table, vectorstore


if __name__ == "__main__":
    main()
