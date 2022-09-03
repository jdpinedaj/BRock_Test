# Some container that is already suitable for unicover
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR '/app/app'
COPY ./Pipfile ./
COPY ./Pipfile.lock ./
# install dependencies, 
# we could probably find an image including this
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile
# RUN pipenv lock -r > requirements.txt
# RUN pip install -r requirements.txt
COPY ./test_stacking.py ./
COPY ./stacking.py ./
COPY ./data ./data
# do tests, usually better to do befor building container, e.g. travis, circelci
RUN python test_stacking.py
CMD ["uvicorn", "stacking:app", "--host", "0.0.0.0", "--port", "80"]