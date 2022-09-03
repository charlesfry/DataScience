from scrapy.http import FormRequest
import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin, urlparse
from datetime import datetime
from scrapy import Request

import json

with open('.\\epartscrawler\\env.json', 'rb') as f:
    creds = json.load(f)
# DOMAIN = creds['DOMAIN']
USERNAME = creds['USERNAME']
PASSWORD = creds['PASSWORD']

DOMAIN = 'shop.jacksoncontrol.com'
ALLOWED_DOMAINS = [DOMAIN]
START_URLS = ['http://' + DOMAIN + '/']
BASE_URL = 'https://shop.jacksoncontrol.com/'

class LoginSpider(scrapy.spiders):
    name = 'login'
    allowed_domains = [DOMAIN]
    start_urls = [BASE_URL]
    def parse(self,response):
        csrf_token = response.xpath('//*[@name=\'csrf_token\']/@value').extract_first()
        yield FormRequest.from_response(response, formdata={'csrf_token': csrf_token, 'user':USERNAME
            , 'pass':PASSWORD}, callback=self.parse_after_login)
    def parse_after_login(self,response):
        pass

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

    # meta = {'cookiejar': response.meta['cookiejar']},

    def start_requests(self):
        return [
            FormRequest(BASE_URL, formdata={"user": USERNAME,
                                                "pass": PASSWORD}, callback=self.parse)]

    def make_requests_from_url(self, url):
        return Request(url, dont_filter=True)

    def parse_func(self, response):
        for link in response.xpath('//a'):
            url = link.xpath('.//@href').get()
            url = urlparse(url).path
            final_url = urljoin(response.url, url)
            yield {
                "loc": final_url,
                "lastmod": datetime.now()
            }


### DONT TOUCH THIS. IT WORKS
#
# with open('.\\epartscrawler\\env.json', 'rb') as f:
#     creds = json.load(f)
# DOMAIN = creds['DOMAIN']
# USERNAME = creds['USERNAME']
# PASSWORD = creds['PASSWORD']
# URL = 'https://' + DOMAIN
#
# DOMAIN = 'jacksoncontrol.com'
#
# ALLOWED_DOMAINS = [DOMAIN]
# START_URLS = ['http://' + DOMAIN + '/']
# BASE_URL = 'https://shop.jacksoncontrol.com/'
#
# class UrlCrawler(CrawlSpider):
#     name = 'url'
#     allowed_domains = ALLOWED_DOMAINS
#     start_urls = START_URLS
#     base_url = BASE_URL
#     rules = [Rule(LinkExtractor(),
#                   callback='parse_func', follow=True)]
#
#     custom_settings = {
#         'CONCURRENT_REQUESTS' : 100,
#         'REACTOR_THREADPOOL_MAXSIZE': 20,
#         'COOKIES_ENABLED': True,
#         'REDIRECT_ENABLED': False,
#         'AJAXCRAWL_ENABLED': True,
#         # 'FEED_URI': 'scrape_' + datetime.now().strftime('%m-%d-%Y-%H-%M-%S') + '_.xml'
#         'ITEM_PIPELINES': {
#             'epartscrawler.pipelines.XmlExportPipeline':100
#         },
#         # 'DOWNLOADER_MIDDLEWARES' : {
#         #     'epartscrawler.middlewares.EpartscrawlerSpiderMiddleware': 543,
#         # }
#     }
#
#     def parse_page(self, response):
#         # do some processing
#         return Request(BASE_URL,
#                               meta={'cookiejar': response.meta['cookiejar']},
#                               # callback=self.parse_other_page
#                        )
#
#     def start_requests(self):
#         for i, url in enumerate(self.start_urls):
#             yield Request("http://www.example.com", meta={'cookiejar': i},
#                                  callback=self.parse_page)
#
#     def parse_func(self, response):
#         for link in response.xpath('//a'):
#             url = link.xpath('.//@href').get()
#             url = urlparse(url).path
#             final_url = urljoin(response.url, url)
#             yield {
#                 "loc": final_url,
#                 "lastmod": datetime.now()
#             }
