from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

browser = webdriver.Firefox()  # can be webdriver.PhantomJS()
browser.get('https://shop.jacksoncontrol.com/Catalog/Products?categoryID=32&productTypeID=5&&')

# wait for the select element to become visible
select_element = WebDriverWait(browser, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.select-size select.sizeOptions")))

select = Select(select_element)
for option in select.options[1:]:
    print(option.text)

browser.quit()