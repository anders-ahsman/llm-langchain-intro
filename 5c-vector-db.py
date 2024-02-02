import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai.embeddings import OpenAIEmbeddings

from common import EXAMPLE_TEXTS

SEARCH_NUM_RESULTS = 3


def main():
    embeddings = OpenAIEmbeddings()

    # Create a LanceDB table
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "my_table",
        data=[
            {
                "vector": embeddings.embed_query("hello world"),
                "text": "hello world",
                "id": "1",
            }
        ],
        mode="overwrite",
    )

    # Create a LanceDB vectorstore and add some texts
    vectorstore = LanceDB(table, embeddings)
    vectorstore.add_texts(EXAMPLE_TEXTS)

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


if __name__ == "__main__":
    main()
