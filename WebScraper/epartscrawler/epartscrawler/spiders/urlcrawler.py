from scrapy.spiders import CrawlSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin, urlparse
from datetime import datetime
from scrapy import Request

import json

with open('.\\epartscrawler\\env.json', 'rb') as f:
    creds = json.load(f)
DOMAIN = creds['DOMAIN']
USERNAME = creds['USERNAME']
PASSWORD = creds['PASSWORD']
BASE_URL = 'https://' + DOMAIN + '/'
START_URLS = [BASE_URL]
ALLOWED_DOMAINS = [DOMAIN]


class UrlCrawler(CrawlSpider):
    name = 'url'
    allowed_domains = ALLOWED_DOMAINS
    start_urls = START_URLS
    base_url = BASE_URL
    rules = [Rule(LinkExtractor(),
                  callback='parse_func', follow=True)]

    custom_settings = {
        'CONCURRENT_REQUESTS' : 100,
        'REACTOR_THREADPOOL_MAXSIZE': 20,
        'COOKIES_ENABLED': True,
        'REDIRECT_ENABLED': False,
        'AJAXCRAWL_ENABLED': True,
        # 'FEED_URI': 'scrape_' + datetime.now().strftime('%m-%d-%Y-%H-%M-%S') + '_.xml'
        'ITEM_PIPELINES': {
            'epartscrawler.pipelines.XmlExportPipeline':100
        },
        # 'DOWNLOADER_MIDDLEWARES' : {
        #     'epartscrawler.middlewares.EpartscrawlerSpiderMiddleware': 543,
        # }
    }

    def parse_page(self, response):
        # do some processing
        return Request(BASE_URL,
                              meta={'cookiejar': response.meta['cookiejar']},
                              # callback=self.parse_other_page
                       )

    def start_requests(self):
        for i, url in enumerate(self.start_urls):
            yield Request("http://www.example.com", meta={'cookiejar': i},
                                 callback=self.parse_page)

    def parse_func(self, response):
        for link in response.xpath('//a'):
            url = link.xpath('.//@href').get()
            url = urlparse(url).path
            final_url = urljoin(response.url, url)
            yield {
                "loc": final_url,
                "lastmod": datetime.now()
            }
