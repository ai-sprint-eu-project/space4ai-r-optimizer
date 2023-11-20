# Copyright 2021 AI-SPRINT

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

FROM python:3.9

# upgrade pip
RUN pip install --upgrade pip

# create working directory
RUN mkdir /webapp
WORKDIR /webapp

# install python requirements
COPY requirements.txt /webapp
RUN pip install -r requirements.txt

# copy app source file
COPY app.py /webapp

# expose port
EXPOSE 3535

# entrypoint
ENTRYPOINT ["python", "app.py"]
