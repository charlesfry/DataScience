import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from datetime import datetime

t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

class JacksoncontrolSpider(CrawlSpider):
    name = 'jacksoncontrol'
    allowed_domains = ['shop.jacksoncontrol.com']
    start_urls = ['http://shop.jacksoncontrol.com/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    custom_settings = {
        'CONCURRENT_REQUESTS': 100,
        'REACTOR_THREADPOOL_MAXSIZE': 20,
        'COOKIES_ENABLED': True,
        'REDIRECT_ENABLED': False,
        'AJAXCRAWL_ENABLED': True,
        # 'FEED_URI': 'scrape_' + datetime.now().strftime('%m-%d-%Y-%H-%M-%S') + '_.xml'
        'ITEM_PIPELINES': {
            'simple.pipelines.XmlExportPipeline': 100
        },
        # 'DOWNLOADER_MIDDLEWARES' : {
        #     'epartscrawler.middlewares.EpartscrawlerSpiderMiddleware': 543,
        # }
    }

    def parse_item(self, response):
        item = {}
        item['loc'] = response.url
        item['lastmod'] = t
        return item