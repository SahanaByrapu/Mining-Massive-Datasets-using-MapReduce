
import requests
import os
import math
import time

url = "http://stateoftheunion.onetwothree.net/texts/index.html"
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

home_page = requests.get(url, headers=headers)

start = home_page.text.find("<br/><br/>\n\n<ul>")
end = home_page.text.find("</ul>\n\n\t\t\t")

links = home_page.text[start: end]
links = links.split("\n")[3:-1]

link_list = [x[13:26] for x in links]

begin = "<h3>"
terminate = '<div class="textNav"><a href="#top">^ Return to top</a></div>'

texts = []
for page in link_list:
  url = "http://stateoftheunion.onetwothree.net/texts/" + page
  content = requests.get(url, headers=headers)
  start = content.text.find(begin)
  end = content.text.find(terminate)
  texts.append(content.text[start:end])



try:
  os.mkdir("texts")
except:
  print("Folder already exists")

for text in texts:
  year = text[text.find(", ") + 2: text.find("</h3>")]
  content = text[text.find("</h3>") + 5:  ]
  f = open("texts/" + year + ".txt", "w")
  f.write(content)
  f.close()

