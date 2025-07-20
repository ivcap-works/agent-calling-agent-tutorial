FROM python:3.13-slim-bookworm
RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY report_writer.py .

# VERSION INFORMATION
ARG VERSION ???
ENV VERSION=$VERSION
ENV PORT=80

# Command to run
ENTRYPOINT ["python",  "/app/report_writer.py"]
