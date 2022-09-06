import scrapy
from scrapy import Request
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin
from scrapy.spiders import CrawlSpider, Rule
from w3lib.url import url_query_cleaner
from datetime import datetime

### trying with scrapy-splash
from scrapy_splash import SplashRequest
WAIT = 4
DOMAIN = 'jacksoncontrol.com'

class UrlSpider(CrawlSpider):
    name = 'url'
    allowed_domains = [DOMAIN]
    start_urls = ['https://shop.' + DOMAIN + '/']
    meta = {'dont_redirect': True, 'handle_httpstatus_list': [301, 302]}
    links=set()
    le = LinkExtractor(allow=r"http[s]?://shop." + DOMAIN, 
                       deny=r"http[s]?://auth.shop." + DOMAIN)
    rules = [Rule(le, 
                  callback='parse', 
                  follow=True,
                  process_request='splash_request'
                  )
             ]        
    
    def start_requests(self):
        for i, url in enumerate(self.start_urls):
            yield SplashRequest(url, self.parse, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': WAIT}
                },
                'cookiejar': i
            })
            
    def splash_request(self, response):
        yield SplashRequest(response.url, self.parse, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': WAIT}
                },
                'cookiejar': response.meta['cookiejar']
            })
    
    def parse(self, response):
        self.links.add(response.url)
        extracted_links = self.le.extract_links(response)
        urls = {u.url for u in extracted_links}
        for url in urls - self.links:
            if url in self.links: continue # double check
            self.parse(url)
            yield SplashRequest(url, self.parse_items, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': WAIT}
                },
                'cookiejar': response.meta['cookiejar']
            })

    
    def parse_items(self, response):
        yield {'loc':url_query_cleaner(response.url),
               'status':response.status,
               'lastedit':datetime.now().strftime("%m/%d/%Y %H:%M:%S")
               }

### I THINK THIS WORKS
'''
class UrlSpider(scrapy.Spider):
    name = 'url'
    allowed_domains = ['jacksoncontrol.com']
    start_urls = ['https://shop.jacksoncontrol.com/']
    meta = {'dont_redirect': True, 'handle_httpstatus_list': [301, 302]}
    links=set()

    # rules = [Rule(LinkExtractor(allow=r"http[s]?://shop.jacksoncontrol.com"), 
    #               callback='parse_item', 
    #               follow=True)]
    le = LinkExtractor(allow=r"http[s]?://shop.jacksoncontrol.com")

    def start_requests(self):
        for url in self.start_urls:
            yield SplashRequest(url, self.parse_items, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': 1.5}
                }
            })
    
    def parse_items(self, response):
        self.links.add(response.url)
        extracted_links = self.le.extract_links(response)
        urls = {u.url for u in extracted_links}
        self.parse(response)
        for url in urls - self.links:
            if url in self.links: continue # double check
            self.links.add(url)
            yield SplashRequest(url, self.parse_items, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': 1.5}
                }
            })

    
    def parse(self, response):
        yield {'loc':response.url,
               'status':response.status,
               'lastedit':time
               }
'''


###
'''
rules = [Rule(LinkExtractor(allow=r"http[s]?://shop.jacksoncontrol.com"), 
                  callback='parse_item', 
                  follow=True)]

    def parse_item(self, response):
        yield scrapy.Request(response.url, self.parse, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': 1.5}
                }
            })
        
    def parse(self, response, **kwargs):
        yield {'loc':response.url,
               'status':response.status,
               'lastedit':time
               }
    
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, self.parse, meta={
                'splash': {
                    'endpoint': 'render.html',
                    'args': {'wait': 1.5}
                }
            })
'''
###

### sept 4 8:45 PM: this works for everything except dynamically loaded urls

# time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
'''
class UrlSpider(CrawlSpider):
    name = 'url'
    allowed_domains = ['jacksoncontrol.com']
    start_urls = ['https://shop.jacksoncontrol.com/']
    meta = {'dont_redirect': True, 'handle_httpstatus_list': [301, 302]}
    links=set()

    rules = [Rule(LinkExtractor(allow=r"http[s]?://shop.jacksoncontrol.com"), 
                  callback='parse_item', 
                  follow=True)]

    def parse_item(self, response):
        yield {'loc':response.url,
               'status':response.status,
               'lastedit':time
               }
'''
    
    

'''
rules = (
    Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
)

def parse_item(self, response):
    item = {}
    #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
    #item['name'] = response.xpath('//div[@id="name"]').get()
    #item['description'] = response.xpath('//div[@id="description"]').get()
    return item
'''