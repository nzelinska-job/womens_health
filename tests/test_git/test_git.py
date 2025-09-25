import requests
import streamlit as st


# Тест без токену (повинен дати 404 для приватного репо)
response = requests.get(
    "https://raw.githubusercontent.com/nzelinska-job/cycle-tracker-data/main/users.csv")
print(f"Without token: {response.status_code}")

# Тест з токеном
headers = {"Authorization": f"token {st.secrets['github_token']}"}
response = requests.get(
    "https://raw.githubusercontent.com/nzelinska-job/cycle-tracker-data/main/users.csv", headers=headers)
print(f"With token: {response.status_code}")

# Тест API
response = requests.get(
    "https://api.github.com/repos/nzelinska-job/cycle-tracker-data/contents/users.csv", headers=headers)
print(f"API: {response.status_code}")
