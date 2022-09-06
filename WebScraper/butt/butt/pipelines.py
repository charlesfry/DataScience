# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface

from scrapy import signals
from scrapy.exporters import XmlItemExporter
from itemadapter import ItemAdapter

class ButtPipeline:
    def process_item(self, item, spider):
        return item

class XmlExportPipeline(object):
    
    def __init__(self):
        self.file = open('urls.xml', 'w+b')
        self.files = {}
        self.exporter = XmlItemExporter(self.file)

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_opened(self, spider):
        self.files[spider] = self.file
        
        self.exporter.start_exporting()

    def spider_closed(self, spider):
        self.exporter.finish_exporting()
        # self.file = self.files.pop(spider)
        self.file.close()
        xml_pipeline(self.file)

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item


import xml.etree.ElementTree as ET
import pickle
def xml_pipeline(f):
    try:
        rows = pickle.load(f)
        to_xml(rows)
    except:
        pass

def to_xml(rows):

    root = ET.Element("urls")

    for row in rows:
        doc = ET.SubElement(root, "url")
        for k,v in row.items():
            # ET.SubElement(doc, k, name="blah").text = "some value1"
            ET.SubElement(doc, k).text = v
    tree = ET.ElementTree(root)
    tree.write("butt.xml")
    
    