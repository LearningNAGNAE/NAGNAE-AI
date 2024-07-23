# /app/models/__init__.py
# 이곳에 필요한 초기화 코드를 추가할 수 있습니다.

# models 패키지를 초기화하는 코드
print("models 패키지가 로드되었습니다.")

# 패키지 내에서 공유할 함수 정의
# def greet(name):
#     return f"Hello, {name}!"

from .grammar_correction import grammar_corrector
from .t2t import t2t