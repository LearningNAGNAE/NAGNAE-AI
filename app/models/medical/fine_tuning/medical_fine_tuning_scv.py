import csv
import re
import logging
from typing import Dict, Any, List

# 상수 정의
QA_PATTERN = re.compile(r'Q\d+:|A\d+:')
INPUT_FILE_PATH = './app/models/medical/fine_tuning/medical_fine_tuning_file.txt'
OUTPUT_FILE_PATH = './app/models/medical/fine_tuning/medical_fine_tuning_file.csv'
FIELDNAMES = ['instruction', 'input', 'output', 'prompt']

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV 생성 관련 함수들
def read_file_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_qa_pairs(content: str) -> List[str]:
    qa_pairs = QA_PATTERN.split(content)
    return [pair.strip() for pair in qa_pairs if pair.strip()]

def create_prompt(question: str, answer: str) -> str:
    return f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n{answer}\n<end_of_turn>"

def process_qa_pairs(qa_pairs: List[str]) -> List[Dict[str, str]]:
    data = []
    for i in range(0, len(qa_pairs), 2):
        if i+1 < len(qa_pairs):
            question = qa_pairs[i]
            answer = qa_pairs[i+1]
            prompt = create_prompt(question, answer)
            data.append({
                'instruction': question,
                'input': '',
                'output': answer,
                'prompt': prompt
            })
    return data

def save_to_csv(data: List[Dict[str, str]], file_path: str) -> None:
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)

def main():
    logger.info("CSV 파일 생성 시작")
    content = read_file_content(INPUT_FILE_PATH)
    qa_pairs = split_qa_pairs(content)
    data = process_qa_pairs(qa_pairs)
    save_to_csv(data, OUTPUT_FILE_PATH)
    logger.info(f"데이터가 {OUTPUT_FILE_PATH} 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
