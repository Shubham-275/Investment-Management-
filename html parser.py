import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_National_Stock_Exchange_of_India"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

company_names = []
for div in soup.select('div.div-col li'):
    company = div.get_text(strip=True)
    company_names.append(company)

with open("nse_company_list.txt", "w", encoding="utf-8") as file:
    for name in sorted(set(company_names)):
        file.write(name + "\n")

print("File saved as nse_company_list.txt")
