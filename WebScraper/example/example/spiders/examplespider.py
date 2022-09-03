from scrapy.spiders import CrawlSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin, urlparse
DOMAIN = 'example.com/'
URL = 'https://' + DOMAIN


class UrlCrawler(CrawlSpider):
    name = 'url'
    allowed_domains = ['books.toscrape.com']
    start_urls = ['http://books.toscrape.com/']
    base_url = 'http://books.toscrape.com/catalogue'
    rules = [Rule(LinkExtractor(allow='books_1/'),
                  callback='parse_func', follow=True)]

    def parse_func(self, response):
        for link in response.xpath('//a'):
            url = link.xpath('.//@href').get()
            url = urlparse(url).path
            final_url = urljoin(response.url, url)
            yield {
                "link": final_url
            }
