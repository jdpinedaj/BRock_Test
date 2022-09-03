# BlackRock ML Challenge

![badge1](https://img.shields.io/badge/language-Python-blue.svg)
![badge2](https://img.shields.io/badge/framework-FastAPI-brightgreen.svg)

## General

This repository is associated with the Challenge described in this [link](https://github.com/jdpinedaj/BRock_Test/tree/master/document).

### To run:

#### Locally

The API can be run locally by executing the following command:

`pip install pipenv`\
`pipenv install --system --deploy --ignore-pipfile`\
`pipenv shell`\
`cd TO/PATH`\
`uvicorn main:app --reload`

#### Using Docker

However, the API can also be run using Docker. To do so, the following commands must be executed:

cd to the folder: `cd PATH/TO/BRock_Test`
run docker: `docker-compose up --build -d`

#### Find Solution at

Recommended: go in browser to http://localhost/docs (Assuming your docker hosts there, if not, check that `docker ps` it's on port 80).\
After that, you can use the API directly with GUI http://localhost/docs

### Files

`stacking.py` holds the API-related code.\
`test_stacking.py` has test cases to validate model is working.\
`Dockerfile` configures the server and installs dependencies. I used the Uvicorn/FastApi in this challenge.\
`docker-compose.yml` simplifies the execution to deploy Docker.\
`create_dataset.py` creates the toy dataset to test the implementation.\
`Pipfile` and `Pipfile.lock` are used to manage dependencies.\
`data` folder contains the toy dataset.\
`model` folder contains the trained model.\
