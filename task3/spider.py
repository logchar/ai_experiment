import csv
from bs4 import BeautifulSoup

for i in range(10):
    for j in range(10):
        for k in range(10):
            page = str(i) + str(j) + str(k)
            with open(f'{page}.html', 'r') as f:
                html_content = f.read()

# 使用BeautifulSoup解析HTML内容
soup = BeautifulSoup(html_content, 'html.parser')


with open('features.csv', 'w', newline='') as csvfile:
    fieldnames = ['feature1', 'feature2', 'feature3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for paragraph in soup.find_all('p'):
        feature1 = paragraph.find('span', {'class': 'description'}).text
        feature2 = paragraph.find('span', {'class': 'achievement'}).text
        feature3 = paragraph.find('span', {'class': 'experience'}).text

        writer.writerow({'feature1': feature1, 'feature2': feature2, 'feature3': feature3})