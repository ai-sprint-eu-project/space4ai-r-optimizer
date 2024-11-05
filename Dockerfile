############################################################################
#                            define base image                             #
############################################################################
FROM ubuntu:20.04 AS base

RUN apt-get dist-upgrade && apt-get update
ARG DEBIAN_FRONTEND=noninteractive TZ="Europe/Rome"

# install modules
RUN apt-get install -y  build-essential \
        		cmake \
        		doxygen \
        		g++ \
        		git \
        		graphviz \
        		libssl-dev \
        		nano \
        		python3-dev \
        		python3-pip \
        		openssl \
        		ssh 
RUN apt-get clean -y

# set working directory
WORKDIR /home/SPACE4AI-R

# upgrade pip and install modules
RUN python3 -m pip install --upgrade pip
COPY python_requirements.txt .
RUN python3 -m pip install --no-cache-dir -r python_requirements.txt

# define parser and logger url
ENV GITLAB=https://gitlab.polimi.it
ENV PARSER_URL=${GITLAB}/ai-sprint/space4ai-parser.git
ENV LOGGER_URL=${GITLAB}/ai-sprint/space4ai-logger.git
ENV PROJECT_ID=776
ENV PARSER_DIR=external/space4ai_parser
ENV LOGGER_DIR=external/space4ai_logger

# install logger
RUN git clone ${LOGGER_URL} ./${LOGGER_DIR}

# load entrypoint code, maximum-workload webapp and dicotomic search code
COPY s4ai-r-opt.py .
COPY maximum_workload.py .
COPY estimate_CPUs.py .

# load optimizer code and make
COPY s4ai-r-optimizer ./s4ai-r-optimizer
RUN mkdir s4ai-r-optimizer/BUILD && \
    cd s4ai-r-optimizer/BUILD && \
    cmake .. && \
    make -j4

############################################################################
#                       build image for development                        #
############################################################################
FROM base AS image-dev

# copy the last change from your brach to invalidate the cache if there 
# was a new change
ADD "${GITLAB}/api/v4/projects/${PROJECT_ID}/repository/branches/main" \
	/tmp/devalidateCache

# install parser (latest version)
RUN git clone ${PARSER_URL} ./${PARSER_DIR}
RUN pip install --no-cache-dir -r ${PARSER_DIR}/requirements.txt

# entrypoint
CMD bash

############################################################################
#                       build image for production                         #
############################################################################
FROM base AS image-prod

# define parser tag
ARG PARSER_TAG=23.12.11

# install parser 
RUN git clone	--depth 1 \
		--branch ${PARSER_TAG} \ 
		${PARSER_URL} \
		./${PARSER_DIR}
RUN pip install --no-cache-dir -r ${PARSER_DIR}/requirements.txt

# entrypoint
CMD bash
