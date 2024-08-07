# 1단계: 데이터 변환 (txt to csv)
import csv
import re

# 파일 경로 설정
file_path = './app/models/law_and_visa/law_and_visa.txt'

# 데이터를 저장할 리스트
data = []

# 텍스트 파일에서 데이터 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 질문과 답변 쌍 추출
qa_pairs = re.findall(r'"([^"]+)","([^"]+)"', content)

for question, answer in qa_pairs:
    # 데이터 형식화
    instruction = question
    input_text = ""  # 이 데이터셋에는 별도의 input이 없음
    output = answer.strip()
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

    # 데이터 리스트에 추가
    data.append({
        'instruction': instruction,
        'input': input_text,
        'output': output,
        'prompt': prompt
    })

# CSV 파일로 저장
csv_filename = './app/models/law_and_visa/gemma_fine_tuning_dataset.csv'
fieldnames = ['instruction', 'input', 'output', 'prompt']

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"데이터가 {csv_filename} 파일로 저장되었습니다.")

# -------------------------------------------------------------------------------------