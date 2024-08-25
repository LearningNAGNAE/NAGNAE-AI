import csv
import json

# 파일 경로 설정
json_file_path = './app/models/job/job_tunning.json'
csv_file_path = './app/models/job/job_fine_tuning_dataset.csv'

# 데이터를 저장할 리스트
data = []

# JSON 파일에서 데이터 읽기
with open(json_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# JSON 데이터 처리
for item in json_data:
    query = item['query']
    for result in item['expected_result']:
        # 데이터 형식화
        instruction = query
        input_text = json.dumps(result, ensure_ascii=False)  # 결과 데이터를 JSON 문자열로 변환
        output = "위 정보를 바탕으로 사용자의 질문에 답변해주세요."
        prompt = f"<start_of_turn>user\n{instruction}\n\n제공된 정보:\n{input_text}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

        # 데이터 리스트에 추가
        data.append({
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'prompt': prompt
        })

# CSV 파일로 저장
fieldnames = ['instruction', 'input', 'output', 'prompt']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"데이터가 {csv_file_path} 파일로 저장되었습니다.")