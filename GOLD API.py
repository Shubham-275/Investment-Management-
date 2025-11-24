import requests
api_url = 'https://api.api-ninjas.com/v1/goldprice'
response = requests.get(api_url, headers={'X-Api-Key': 'ZPgVF+ByP8seMF8bKToPMg==YDkPQi5cuKIhP730'})
if response.status_code == requests.codes.ok:
    print(response.text)
else:
    print("Error:", response.status_code, response.text)
