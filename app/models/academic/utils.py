from config import openai, cross_encoder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import torch  # torch import 추가

class Entities(BaseModel):
    universities: List[str] = Field(default_factory=list, description="List of university names")
    majors: List[str] = Field(default_factory=list, description="List of major names")
    regions: List[str] = Field(default_factory=list, description="List of regions")
    school_types: List[str] = Field(default_factory=list, description="List of school types")
    est_types: List[str] = Field(default_factory=list, description="List of establishment types (e.g., 국립, 사립)")
    campus_names: List[str] = Field(default_factory=list, description="List of campus names")
    keywords: List[str] = Field(default_factory=list, description="List of other relevant keywords")


def detect_language(text: str) -> str:
    """언어 감지 함수"""
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    return response.content.strip().lower()

def korean_language(text: str) -> str:
    """한국어로 번역하는 함수"""
    system_prompt = "You are a translation expert. Your task is to detect the language of a given text and translate it into Korean. Please provide only the translated text in Korean, without any additional explanations or information."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    return response.content.strip().lower()

def english_language(text: str) -> str:
    """한국어로 번역하는 함수"""
    system_prompt = (
        "You are a translation expert with a focus on accuracy, especially when handling place names, proper nouns, and similar-sounding terms. "
        "Your task is to detect the language of a given text and translate it into English. "
        "Ensure that place names (e.g., Cheongju, Chungju) and proper nouns are translated accurately without confusion. "
        "Please provide only the translated text in English, without any additional explanations or information."
    )
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    print(response.content.strip().lower());
    return response.content.strip().lower()

def trans_language(text: str, target_language: str) -> str:
    """텍스트를 감지된 언어로 번역하는 함수"""
    system_prompt = f"You are a professional translator. Translate the following text into {target_language} accurately and concisely. Provide only the translated text without any additional comments or explanations."
    human_prompt = f"Translate this text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    return response.content.strip()


def extract_entities(query: str) -> Entities:
    try:
        parser = PydanticOutputParser(pydantic_object=Entities)
        prompt = f"""
        Extract relevant entities from the following query. The entities should include:
        - University names
        - Major names
        - Regions
        - School types (e.g., 4년제, 전문대학)
        - Establishment types (e.g., 국립, 사립)
        - Campus names
        - Other relevant keywords

        Query: {query}

        After extracting the entities, translate the results into English.

        {parser.get_format_instructions()}
        """
        messages = [
            {"role": "system", "content": "You are an expert in extracting relevant entities from text."},
            {"role": "user", "content": prompt}
        ]
        response = openai.invoke(messages)

        print(response.content)

        return parser.parse(response.content)
    except Exception as e:
        print(f"엔티티 추출 중 오류 발생: {e}")
        return Entities()  # 이렇게 하면 모든 필드가 빈 리스트로 초기화됩니다.

def generate_elasticsearch_query(entities: Entities):
    should_clauses = []
    if entities.universities:
        should_clauses.append({"terms": {"metadata.schoolName.keyword": entities.universities}})
    if entities.majors:
        should_clauses.append({"terms": {"metadata.major.keyword": entities.majors}})
    if entities.regions:  # 여기를 'regions'로 수정
        should_clauses.append({"terms": {"metadata.region.keyword": entities.regions}})
    if entities.school_types:
        should_clauses.append({"terms": {"metadata.schoolType.keyword": entities.school_types}})
    if entities.est_types:
        should_clauses.append({"terms": {"metadata.estType.keyword": entities.est_types}})
    if entities.campus_names:
        should_clauses.append({"terms": {"metadata.campusName.keyword": entities.campus_names}})
    if entities.keywords:
        should_clauses.append({
            "multi_match": {
                "query": " ".join(entities.keywords),
                "fields": [
                    "text", "metadata.schoolName", "metadata.major", "metadata.region", 
                    "metadata.schoolType", "metadata.estType", "metadata.campusName", 
                    "metadata.summary^2", "metadata.job", "metadata.qualifications",
                    "metadata.lClass", "metadata.mClass", "metadata.facilName", "metadata.part",
                    "metadata.majorName"
                ],
                "type": "best_fields",
                "tie_breaker": 0.3
            }
        })
    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
    }

def generate_response_with_fine_tuned_model(prompt, model, tokenizer, max_length=100):
    print(f"Prompt type: {type(prompt)}")
    print(f"Prompt content: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')  # 입력 데이터를 GPU로 이동
    outputs = model.generate(**inputs, max_new_tokens=256, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)