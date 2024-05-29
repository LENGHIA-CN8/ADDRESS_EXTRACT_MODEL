import scrapy
from scrapy.selector import Selector
import json
import re
from bs4 import BeautifulSoup
from address_crawl.items import AddressCrawlItem

def striphtml_crawled(data, DEBUG=False, url = ''):
   
    try:
        assert isinstance(data, str), 'Not String'
    except:
        print(data)
    p = re.compile(r'<.*?>')
    p = p.sub(' ', data)
    p = re.sub('[ \t]+',' ', p)
    p = re.sub(r'\n+', '\n', p)
    p = re.sub('&nbsp;','', p)
    p = re.sub('(&ldquo;|&rdquo;|&hellip;|&quot;)', '', p)
    p = p.replace("&amp;", " ")
    p = re.sub(r'!\[\]\([^)]+\)', '', p, flags=re.MULTILINE)
    p = '\n'.join(line.strip() for line in p.split('\n'))
    return p

class TopicLinkCrawlerSpider(scrapy.Spider):
    name = 'address_crawl'
    allowed_domains = ["www.tratencongty.com"]

    start_urls = 'https://www.tratencongty.com/?page={}'
    
    def start_requests(self):
        for idx in range(26167,32666):
            yield scrapy.Request(self.start_urls.format(idx), headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
    
    def parse(self, response):
        company_tags = response.css('div.search-results').getall()
        for tag in company_tags:
            soup = BeautifulSoup(tag, 'html.parser')
            # Find the company name within the 'a' tag
            company_name = soup.find('a').text
            infor = soup.find('p').text
            match = re.search("địa chỉ:", infor.lower())
            if match:
                st_id = match.start()
                # print(st_id)
                # print(company_name)
                # print('meta', infor[:st_id].strip())
                # print('add', infor[st_id:].strip())
                company_info = AddressCrawlItem()
                company_info['company_name'] = company_name
                company_info['meta_data'] = infor[:st_id].strip()
                company_info['address'] = infor[st_id:].strip()
                yield company_info
            else:
                return

    
