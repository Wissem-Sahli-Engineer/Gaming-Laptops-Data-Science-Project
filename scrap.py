import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# 1. Setup Browser
chrome_options = Options()
# chrome_options.add_argument("--headless") # Uncomment to run without a window
driver = webdriver.Chrome(options=chrome_options)

all_laptops = []

try:
    # Based on your calculation: 296 items / 24 per page = 13 pages
    for k in range(1, 14):
        url = f"https://www.tunisianet.com.tn/681-pc-portable-gamer?page={k}"
        print(f"--- Working: Accessing Page {k} ---")
        
        try:
            driver.get(url)
            time.sleep(5) # Wait for dynamic price/stock to load

            # Find all product containers
            products = driver.find_elements(By.CLASS_NAME, "wb-product-desc")

            for product in products:
                try:
                    # Extracting data based on your HTML mapping
                    name = product.find_element(By.CLASS_NAME, "product-title").text            
                    ref = product.find_element(By.CLASS_NAME, "product-reference").text 
                    desc = product.find_element(By.CLASS_NAME, "listds").text 
                    price = product.find_element(By.CLASS_NAME, "price").text 
                    
                    all_laptops.append({
                        "Name": name,
                        "Reference": ref,
                        "Description": desc,
                        "Price": price
                    })
                except Exception as e:
                    print(f"Skipping an item on page {k} due to missing data.")

        except Exception as e:
            print(f"!!! Error: Failed to load Page {k} !!!")

    # 2. Convert to DataFrame
    df = pd.DataFrame(all_laptops)
    print("--- Success: Data scraping complete! ---")
    print(df.head())

    # Optional: Save to Excel/CSV
    df.to_csv("gaming_laptops.csv", index=False)

finally:
    driver.quit()