from typing import Literal

from pydantic import BaseModel, Field


class Person(BaseModel):
    age: int
    workclass: Literal[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
    ]
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th",
    ]
    maritalStatus: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ] = Field(alias="marital-status")
    occupation: Literal[
        "Adm-clerical",
        "Exec-managerial",
        "Handlers-cleaners",
        "Prof-specialty",
        "Other-service",
        "Sales",
        "Transport-moving",
        "Farming-fishing",
        "Machine-op-inspct",
        "Tech-support",
        "Craft-repair",
        "Protective-serv",
        "Armed-Forces",
        "Priv-house-serv",
    ]
    relationship: Literal[
        "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
    ]
    race: Literal["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    sex: Literal["Male", "Female"]
    hoursPerWeek: int = Field(alias="hours-per-week")
    nativeCountry: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
    ] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 23,
                "workclass": "Private",
                "education": "11th",
                "marital-status": "Married-civ-spouse",
                "occupation": "Transport-moving",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


class InferenceResponse(BaseModel):
    salary: Literal["<=50K", ">50K"]

    class Config:
        schema_extra = {"example": {"salary": "<=50K"}}
