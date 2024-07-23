conda create -n nagnae-ai python=3.10

conda acivate nagnae-ai

conda install cudatoolkit

pip install insightface
※사용할려면 'vs_BuildTools.exe'를 설치해야한다.

pip install transformers

pip install fastapi

pip install "uvicorn[standard]"

uvicorn main:app

pip install mysql-connector-python sqlalchemy




















pip install -r requirements.txt //라이브러리가 모두 설치된다.