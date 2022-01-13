import subprocess
import requests #install requests in docker

process = subprocess.Popen("python bots-webserver.py")

# go to login page to get csrf cookie
url = "http://localhost:8080/login"
payload = {}
headers = {}
response = requests.request("GET", url, headers=headers, data=payload)
csrfCookieValue = response.cookies.get("csrftoken")




# login with csrf cookie and get sessionid cookie
url = "http://localhost:8080/login"
payload = {
    "username": "bots",
    "password": "bots",
    "csrfmiddlewaretoken": csrfCookieValue,
    "next": ""
}
headers = {
  'Cookie': 'csrftoken=' + csrfCookieValue
}
response = requests.request("POST", url, headers=headers, data=payload)
sessionCookieValue = response.cookies.get("sessionid")



# upload the zip file to bots
url = "http://localhost:8080/plugin"
payload = {
    "submit" : ""
}
files = {'file': ('alps.zip', open('./alps.zip','rb'), 'application/zip')}
headers = {
  'Cookie': 'csrftoken=' + csrfCookieValue + '; sessionid=' + sessionCookieValue
}
response = requests.request("POST", url, headers=headers, data=payload, files=files)


process.kill()