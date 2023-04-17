# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:15:01 2022

@author: Guillermo Pizana
"""
import json 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bf
import pandas as pd
import re


from string import digits




def extract_texts(driver, url):
    
    driver.get(url)
    
    sonet_list = driver.find_elements(By.CSS_SELECTOR,'.normalsans')
    
    sonet_texts = []
    
    for sonet in sonet_list[1:]:
    
        sonet_texts.append(sonet_cleaner(sonet.text))
    
    return sonet_texts


def sonet_cleaner(sonet):

    remove_digits = str.maketrans('', '', digits)
    
    clean_sonet = sonet.translate(remove_digits)
    
    clean_sonet = clean_sonet.lower()
    
    clean_sonet = clean_sonet.replace('\n\n','')
    
    clean_sonet = clean_sonet.replace('    ','')
    
    #clean_sonet = " ".join(clean_sonet.split())
    
    return clean_sonet



def write_file(sonet_texts):
    
    with open('sonets.txt', 'w') as f:
    
        for sonet in sonet_texts:
            
            f.write(sonet)
            
        f.close()
    
    
    return




def crawl():
    
    driver = webdriver.Chrome(ChromeDriverManager().install())
    
    url = 'https://www.opensourceshakespeare.org/views/sonnets/sonnet_view.php?range=viewrange&sonnetrange1=1&sonnetrange2=154'
    
    
    sonet_list = extract_texts(driver, url)
    
    write_file(sonet_list)
    
    
    driver.close()
    
    
    return