import sys
import asyncio

from .article import Article
from .jina_client import JinaClient
from .readability_extractor import ReadabilityExtractor


class Crawler:
    async def crawl(self, url: str) -> Article:
        # To help LLMs better understand content, we extract clean
        # articles from HTML, convert them to markdown, and split
        # them into text and image blocks for one single and unified
        # LLM message.
        #
        # Jina is not the best crawler on readability, however it's
        # much easier and free to use.
        #
        # Instead of using Jina's own markdown converter, we'll use
        # our own solution to get better readability results.
        jina_client = JinaClient()
        html = await asyncio.to_thread(jina_client.crawl, url, return_format="html")
        extractor = ReadabilityExtractor()
        article = await asyncio.to_thread(extractor.extract_article, html)
        article.url = url
        return article


if __name__ == "__main__":
    if len(sys.argv) == 2:
        url = sys.argv[1]
    else:
        url = "https://fintel.io/zh-hant/s/br/nvdc34"
    
    async def main():
        crawler = Crawler()
        article = await crawler.crawl(url)
        print(article.to_markdown())

    asyncio.run(main())
