from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set this to your file manager URL
URL = "http://localhost:18910/#/files"

# Path to your chromedriver, or ensure it's in PATH
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Remove this if you want to see the browser

driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
driver.get(URL)

try:
    # Wait for the main file manager element to appear
    main = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "main"))
    )
    print("Main file manager loaded.")

    # Screenshot for debugging
    driver.save_screenshot("file_manager_screenshot.png")
    print("Screenshot saved as file_manager_screenshot.png")

    # Check for error messages
    errors = driver.find_elements(By.CSS_SELECTOR, ".text-red-300, .text-red-500")
    if errors:
        print("Errors found:")
        for e in errors:
            print("-", e.text)
    else:
        print("No visible error messages.")

    # Check for file/folder rows
    rows = driver.find_elements(By.CSS_SELECTOR, ".fm-tree-row, .fm-file-row")
    print(f"Found {len(rows)} file/folder rows.")
    if not rows:
        print("No files or folders are visible in the UI.")

finally:
    driver.quit()
