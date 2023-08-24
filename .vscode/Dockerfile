FROM python:alpine
# required for `pip install psutil`
RUN apk update && apk add python3-dev gcc linux-headers musl-dev
# required for vscode
RUN apk update && apk add git
# project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
# enable devcontainer updateRemoteUserUID
RUN adduser -SD --shell /bin/sh vscode
USER vscode
ENV PATH=/home/vscode/.local/bin:$PATH
