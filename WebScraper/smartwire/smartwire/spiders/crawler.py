import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy import Request
from smartwire.items import SmartwireItem
import json

import scrapy

class AwesomeSpiderWithPage(scrapy.Spider):
    name = "url"

    def start_requests(self):
        yield scrapy.Request(
            url="https://example.org",
            callback=self.parse_first,
            meta={"playwright": True, "playwright_include_page": True},
            errback=self.errback_close_page,
        )

    def parse_first(self, response):
        page = response.meta["playwright_page"]
        return scrapy.Request(
            url="https://example.com",
            callback=self.parse_second,
            meta={"playwright": True, "playwright_include_page": True, "playwright_page": page},
            errback=self.errback_close_page,
        )

    async def parse_second(self, response):
        page = response.meta["playwright_page"]
        title = await page.title()  # "Example Domain"
        await page.close()
        return {"title": title}

    async def errback_close_page(self, failure):
        page = failure.request.meta["playwright_page"]
        await page.close()
'''
class CrawlerSpider(CrawlSpider):
    name = 'url'
    allowed_domains = ['smartwire.com']
    allowed_domains = ['shitonmyballs.com']
    start_urls = ['https://www.smartwire.com']
    start_urls = ["https://www.shitonmyballs.com"]

    # allowed_domains = ['https://doc.scrapy.org/']
    # start_urls = ['https://doc.scrapy.org/en/latest/topics/spiders.html#crawling-rules']
    le = LinkExtractor()
    rules = (
        Rule(le, callback='parse_item', follow=False, process_request='set_playwright_true'),
    )
    custom_settings = {
        'DOMAIN_DEPTHS' : {'example.python-scraping.com': 2},
        'DOWNLOAD_HANDLERS' : {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
    }

    # suggested by stackoverflow
    def set_playwright_true(self, response):
        response.meta["playwright"] = True
        return response

    def start_requests(self):
        self.start_urls = [self.start_urls[0]]
        for url in self.start_urls: # for now, make sure there is only 1 starting domain
            yield scrapy.Request(
                url=url,
                meta={"playwright": True,
                      # "playwright_include_page": True
                      },
                # callback=self.parse,
                # errback=self.errback_close_page
            )
            return super().start_requests()
    def parse_first(self, response):
        page = response.meta['playwright_page']
        return Request(
            url=response.url,
            callback=self.parse,
            meta={"playwright": True, "playwright_include_page": True, "playwright_page": page},
            errback=self.errback_close_page
        )

    async def parse(self, response, **kwargs):
        page = response.meta["playwright_page"]
        yield scrapy.Request(
            url=response.url,
            callback=self.parse_item,
            meta={"playwright": True, "playwright_include_page": True, "playwright_page": page},
        )

    async def parse_item(self, response, **kwargs):
        page = response.meta['playwright_page']
        item = SmartwireItem()
        item['url'] = response.url
        # await page.close()
        # # item['productid'] = response.xpath('/html/body/div[3]/div/div[12]/div/div/div/div/div[1]/h1/text()').extract()
        # yield item


    async def errback_close_page(self, failure):
        page = failure.request.meta['playwright_page']
        await page.close()

'''









"""
    name = 'crawler'
    DOMAIN = 'smartwire.com'
    allowed_domains = [DOMAIN]
    base_url = f'https://www.{DOMAIN}/'
    # start_urls = [f'https://www.smartwire.com/']
    start_urls = [f'https://www.smartwire.com/products/detail/031201/']
    le = LinkExtractor(allow=f'products/.*')
    rules = [Rule(le,
                  callback='parse_item',
                  follow=True,
                  )
             ]

    def parse_item(self, response):
        item = {}
        item['url'] = response.url
        item['productid'] = response.xpath('/html/body/div[3]/div/div[12]/div/div/div/div/div[1]/h1').get()
        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()

        yield item
    """