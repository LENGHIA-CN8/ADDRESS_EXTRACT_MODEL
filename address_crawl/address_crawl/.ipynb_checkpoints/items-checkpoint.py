# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class AddressCrawlItem(scrapy.Item):
    # define the fields for your item here like:
    company_name = scrapy.Field()
    meta_data = scrapy.Field()
    address = scrapy.Field()
    
