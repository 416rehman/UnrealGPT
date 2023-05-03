# UnrealGPT

This is a LangChain project with over 1700+ pages of documentation scraped from https://docs.unrealengine.com/5.1/en-US/

Data is scraped using BeautifulSoup and stored in the `data\scraping_cache` folder. If the data is already scraped, it will be loaded from the cache instead of scraping again to save time.

FAISS is used as the vector store to store the OpenAI embeddings. The FAISS index is stored in the `data` folder with the `.index` extension. If the index is already built, it will be loaded from the cache instead of building again to save time.

*To start fresh (scrapping and building the index again), delete the `data` folder.*

## How to run
All commands are run from the root of the repository.

1. Clone this repository.
2. Run `pip install -r requirements.txt` to install the dependencies.
3. Set the `OPENAI_API_KEY` environment variable to your OpenAI API key on your system.

4. Install [langchain-serve](https://github.com/jina-ai/langchain-serve) then run `lc-serve deploy local main`.

Visit `http://localhost:8080/docs` to interact with the API. (Make sure to provide the `OPENAI_API_KEY` in the request)

![img.png](https://i.imgur.com/inS8Fen.png)

