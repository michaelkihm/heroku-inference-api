import requests

inf_endpoint = "https://inferenceapi0.herokuapp.com/inference"
request_body = {
    "age": 23,
    "workclass": "Private",
    "education": "11th",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Transport-moving",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States",
}
print(f"POST to {inf_endpoint}")
res = requests.post(inf_endpoint, json=request_body)
res_json = res.json()

assert res.status_code == 200, "Could not fetch data from heroku api"
assert res_json["salary"] in [
    "<=50K",
    ">50K",
], "JSON response from heroku api in invalid"

print(f"response body: {res_json} with status code {res.status_code}")
