FROM python:3.10-slim

RUN apt-get dist-upgrade && apt-get update
RUN apt-get install -y  build-essential \
                        git \
                        graphviz \
                        vim
RUN apt-get clean -y

# set working directory
WORKDIR /home/SPACE4AI-R-utilities

# upgrade pip and install modules
RUN python3 -m pip install --upgrade pip
COPY python_requirements.txt .
RUN python3 -m pip install --no-cache-dir -r python_requirements.txt

# define logger url
ENV GITLAB=https://gitlab.polimi.it
ENV LOGGER_URL=${GITLAB}/ai-sprint/space4ai-logger.git
ENV PROJECT_ID=776
ENV LOGGER_DIR=external/space4ai_logger

# install logger
RUN mkdir ./external
RUN git clone ${LOGGER_URL} ./${LOGGER_DIR}

# load utilities code
COPY . .

# entrypoint
CMD bash
