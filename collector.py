from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from PIL import Image
import io
import ast
import numpy as np
import requests

class DataCollector:
    pass

class UnsplashCollector(DataCollector):
    def __init__(self, url_map, driver_path):
        self.url_map = url_map
        self.train_data = []
        self.driver_path = driver_path
        self.classify_map = {}
        self.test_data = []
        self.link_dict = {}

    def collect(self):
        self.init_classify_map()
        for key in self.url_map:
            driver = webdriver.Chrome(self.driver_path)
            classification_label = key
            url = self.url_map[key]
            driver.get(url)
            print(f'Visiting {classification_label} gallery...')
            button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='_3sS4m']/button"))
            )
            driver.execute_script("arguments[0].click();", button)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            driver.execute_script("window.scrollTo(document.body.scrollHeight, document.body.scrollHeight/2)")
            time.sleep(2)
            driver.execute_script("window.scrollTo(document.body.scrollHeight/2, document.body.scrollHeight)")
            time.sleep(2)
            self.get_hrefs(driver.page_source, key)
            driver.close()
        for key in self.link_dict:
            l = len(self.link_dict[key])
            print(f'{key}s found: {l}')
        
        desired = input("""Please enter a tuple of quantities desired in the order the keys were listed.\nE.g., (100,150) for 100 images from the first category and 150 images from the second\n""")

        res = self.translate(desired)
        return res

    def translate(self, desired):
        try:
            trans = ast.literal_eval(desired) 
            if isinstance(trans, int):
                try_again = input("Invalid input, try again: ")
                self.translate(try_again)
            if isinstance(trans, tuple) and len(trans) == len(self.link_dict.keys()):
                for (key, limit) in zip(self.link_dict.keys(), trans):
                    print(f'{limit} {key}s desired')
                ans = input('continue to download? (y/n)')
                if ans == 'y':
                    for (key, limit) in zip(self.link_dict.keys(), trans):
                        n = 0
                        for index, link in enumerate(self.link_dict[key]):
                            try:
                                r = requests.get(link)
                                image = Image.open(io.BytesIO(r.content)).resize((32, 32)).convert('RGB')
                                d = np.asarray(image)
                                d = np.ravel(d)/255
                                categories = len(list(self.classify_map.keys()))
                                e = np.zeros((categories,))
                                e[self.classify_map[key]] = 1.0
                                self.train_data.append((d, e))
                                print(f'{n+1} {key} downloaded.')
                                n+=1
                                if n == limit:
                                    break
                            except Exception as e:
                                print(e)
                return np.array(self.train_data)
            
        except ValueError:
            try_again = input("Invalid input, try again: ")
            self.translate(try_again)

    def get_hrefs(self, source, key):
        self.link_dict[key] = []
        soup = BeautifulSoup(source, 'html.parser')
        for link in soup.find_all('a'):
            cont = link.get('href')
            if '/photos/' in cont:
                if 'download?force=true' in cont:
                    self.link_dict[key].append(cont)
                else:
                    self.link_dict[key].append(f'https://unsplash.com{cont}/download?force=true')
        self.link_dict[key] = list(set(self.link_dict[key]))

    def init_classify_map(self):
        i = 0
        for key in self.url_map:
            self.classify_map[key] = i
            i+=1 


