import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from datetime import datetime
from scrapy.http import FormRequest

t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

class JacksoncontrolSpider(CrawlSpider):
    name = 'jacksoncontrol'
    allowed_domains = ['shop.jacksoncontrol.com']
    start_urls = ['http://shop.jacksoncontrol.com/']

    rules = (
        # Extract links matching 'category.php' (but not matching 'subsection.php')
        # and follow links from them (since no callback means follow=True by default).
        Rule(LinkExtractor(allow=('category\.php',), deny=('subsection\.php',))),

        # Extract links matching 'item.php' and parse them with the spider's method parse_item
        Rule(LinkExtractor(), callback='parse_item'),
    )

    def parse_item(self, response):
        # item = scrapy.Item()
        # item['loc'] = response.url
        url = response.xpath('//a/@href')[0].get()
        yield response.follow(url, self.parse_additional_page, cb_kwargs={'loc':url})

    def parse_additional_page(self, response, item):
        item['additional_data'] = response.xpath('//p[@id="additional_data"]/text()').get()
        return item