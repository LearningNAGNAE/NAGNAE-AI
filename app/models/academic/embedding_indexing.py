import json
import time
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import es_client, embedding, BATCH_SIZE, RATE_LIMIT_DELAY, pdf_path
from data_collection import fetch_university_data, fetch_university_major, fetch_major_details, load_pdf_document

def process_in_batches(data, process_func):
    """데이터를 배치로 처리하는 함수"""
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i+BATCH_SIZE]
        process_func(batch)
        time.sleep(RATE_LIMIT_DELAY)

def index_exists(index_name):
    """인덱스가 이미 있는지 확인하는 함수"""
    return es_client.indices.exists(index=index_name)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_university_data():
    if index_exists('university_data'):
        print("University data index already exists. Skipping...")
        return
    data = fetch_university_data()
    def process_batch(batch):
        for item in batch:
            text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
            vector = embedding.embed_query(text)
            metadata = {
                'source': 'university_data',
                'schoolName': item['schoolName'],
            }
            
            # 선택적 필드들을 메타데이터에 추가
            optional_fields = ['campusName', 'collegeInfoUrl', 'schoolType', 'link', 'schoolGubun', 
                               'adres', 'region', 'totalCount', 'estType', 'seq']
            for field in optional_fields:
                if field in item:
                    metadata[field] = item[field]
            
            doc = {
                'text': text,
                'vector': vector,
                'metadata': metadata
            }
            es_client.index(index='university_data', body=doc)

    process_in_batches(data['dataSearch']['content'], process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_university_major():
    if index_exists('university_major'):
        print("University major index already exists. Skipping...")
        return
    major_data = fetch_university_major()
    if 'dataSearch' not in major_data or 'content' not in major_data['dataSearch']:
        print("Unexpected data structure in university major data")
        return
    
    def process_batch(batch):
        for item in batch:
            text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
            vector = embedding.embed_query(text)
            metadata = {
                'source': 'university_major'
            }
            
            # 선택적 필드들을 메타데이터에 추가
            optional_fields = ['lClass', 'facilName', 'majorSeq', 'mClass', 'totalCount']
            for field in optional_fields:
                if field in item:
                    metadata[field] = item[field]
            
            doc = {
                'text': text,
                'vector': vector,
                'metadata': metadata
            }
            es_client.index(index='university_major', body=doc)

    process_in_batches(major_data['dataSearch']['content'], process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_major_details():
    if index_exists('major_details'):
        print("Major details index already exists. Skipping...")
        return
    major_data = fetch_university_major()
    if 'dataSearch' not in major_data or 'content' not in major_data['dataSearch']:
        print("Unexpected data structure in university major data")
        return
    
    major_seqs = [item['majorSeq'] for item in major_data['dataSearch']['content']]
    
    def process_batch(batch):
        for major_seq in batch:
            major_data = fetch_major_details(major_seq)
            if 'dataSearch' in major_data and 'content' in major_data['dataSearch'] and major_data['dataSearch']['content']:
                item = major_data['dataSearch']['content'][0]  # Assuming one item per major_seq
                text = f"{item.get('major', '')} {item.get('summary', '')}"
                vector = embedding.embed_query(text)
                metadata = {
                    'source': 'major_details'
                }
                
                # 선택적 필드들을 메타데이터에 추가
                optional_fields = ['major', 'salary', 'employment', 'department', 'summary', 
                                   'job', 'qualifications', 'interest', 'property']
                for field in optional_fields:
                    if field in item:
                        metadata[field] = item[field]
                
                # 리스트 형태의 데이터는 문자열로 변환하여 저장
                list_fields = ['relate_subject', 'career_act', 'enter_field', 'main_subject', 'chartData']
                for field in list_fields:
                    if field in item:
                        metadata[field] = json.dumps(item[field])
                
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': metadata
                }
                es_client.index(index='major_details', body=doc)
            else:
                print(f"No data found for major_seq: {major_seq}")

    process_in_batches(major_seqs, process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_pdf_data():
    if index_exists('pdf_data'):
        print("PDF data index already exists. Skipping...")
        return
    documents = load_pdf_document(pdf_path)
    def process_batch(batch):
        for doc in batch:
            vector = embedding.embed_query(doc.page_content)
            es_doc = {
                'text': doc.page_content,
                'vector': vector,
                'metadata': {
                    'source': 'pdf',
                    'page': doc.metadata['page']
                }
            }
            es_client.index(index='pdf_data', body=es_doc)

    process_in_batches(documents, process_batch)

# 업데이트(임베딩 및 인덱스)
def update_university_data():
    new_data = fetch_university_data()
    for item in new_data['dataSearch']['content']:
        query = {
            "query": {
                "match": {
                    "metadata.seq": item['seq']
                }
            }
        }
        result = es_client.search(index="university_data", body=query)

        if result['hits']['total']['value'] == 0:
            # 새로운 데이터 추가
            text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
            vector = embedding.embed_query(text)
            doc = {
                'text': text,
                'vector': vector,
                'metadata': item
            }
            es_client.index(index='university_data', body=doc)
        else:
            # 기존 데이터 업데이트
            existing_doc = result['hits']['hits'][0]
            if existing_doc['_source']['metadata'] != item:
                text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': item
                }
                es_client.update(index='university_data', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_university_major():
    new_data = fetch_university_major()
    for item in new_data['dataSearch']['content']:
        query = {
            "query": {
                "match": {
                    "metadata.majorSeq": item['majorSeq']
                }
            }
        }
        result = es_client.search(index="university_major", body=query)

        if result['hits']['total']['value'] == 0:
            # 새로운 데이터 추가
            text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
            vector = embedding.embed_query(text)
            doc = {
                'text': text,
                'vector': vector,
                'metadata': item
            }
            es_client.index(index='university_major', body=doc)
        else:
            # 기존 데이터 업데이트
            existing_doc = result['hits']['hits'][0]
            if existing_doc['_source']['metadata'] != item:
                text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': item
                }
                es_client.update(index='university_major', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_major_details():
    major_data = fetch_university_major()
    for item in major_data['dataSearch']['content']:
        major_seq = item['majorSeq']
        major_details = fetch_major_details(major_seq)
        if 'dataSearch' in major_details and 'content' in major_details['dataSearch'] and major_details['dataSearch']['content']:
            detail_item = major_details['dataSearch']['content'][0]
            query = {
                "query": {
                    "match": {
                        "metadata.major": detail_item['major']
                    }
                }
            }
            result = es_client.search(index="major_details", body=query)

            if result['hits']['total']['value'] == 0:
                # 새로운 데이터 추가
                text = f"{detail_item.get('major', '')} {detail_item.get('summary', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': detail_item
                }
                es_client.index(index='major_details', body=doc)
            else:
                # 기존 데이터 업데이트
                existing_doc = result['hits']['hits'][0]
                if existing_doc['_source']['metadata'] != detail_item:
                    text = f"{detail_item.get('major', '')} {detail_item.get('summary', '')}"
                    vector = embedding.embed_query(text)
                    doc = {
                        'text': text,
                        'vector': vector,
                        'metadata': detail_item
                    }
                    es_client.update(index='major_details', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_pdf_data():
    # PDF 데이터는 일반적으로 자주 변경되지 않으므로, 
    # 파일이 변경된 경우에만 업데이트하는 로직을 구현할 수 있습니다.
    # 예를 들어, 파일의 수정 날짜를 확인하여 변경된 경우에만 처리할 수 있습니다.
    pass

# embed_and_index_university_major, embed_and_index_major_details, embed_and_index_pdf_data 함수도 유사한 방식으로 구현
# 인덱스 업데이트에 필요한 함수들
def update_indices():
    print("Updating indices...")

    # 대학 정보 업데이트
    update_university_data()

    # 대학 전공 정보 업데이트
    update_university_major()

    # 전공 상세 정보 업데이트
    update_major_details()

    # PDF 데이터 업데이트 (필요한 경우)
    update_pdf_data()

    print("Indices update completed.")
