from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from functools import partial
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import re
from konlpy.tag import Kkma
from typing import List, Dict
from langchain_community.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever as ElasticsearchRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import BaseRetriever, Document
from typing import List
from contextlib import asynccontextmanager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import helpers  # bulk 작업을 위해 필요합니다
from elasticsearch import Elasticsearch  # 이 줄을 추가합니다
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

agent_executor = None
es_client = None
es_retriever = None
embeddings = OpenAIEmbeddings()


app = FastAPI()

class Query(BaseModel):
    input: str
    session_id: str

pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# Session memories storage
session_memories = {}

# Initialize KoNLPy
kkma = Kkma()

# 지역명 사전 정의
region_dict = {
    "서울": [
        ["서울", "서울시", "서울특별시"],
        ["Seoul", "Seoul City", "Seoul Special City"],
        ["首尔", "首尔市", "首尔特别市"],
        ["ソウル", "ソウル市", "ソウル特別市"]
    ],
    "경기": [
        ["경기", "경기도"],
        ["Gyeonggi", "Gyeonggi Province"],
        ["京畿", "京畿道"],
        ["キョンギ", "キョンギ道"]
    ],
    "인천": [
        ["인천", "인천시", "인천광역시"],
        ["Incheon", "Incheon City", "Incheon Metropolitan City"],
        ["仁川", "仁川市", "仁川广域市"],
        ["インチョン", "インチョン市", "インチョン広域市"]
    ],
    "부산": [
        ["부산", "부산시", "부산광역시"],
        ["Busan", "Busan City", "Busan Metropolitan City"],
        ["釜山", "釜山市", "釜山广域市"],
        ["プサン", "プサン市", "プサン広域市"]
    ],
    "대구": [
        ["대구", "대구시", "대구광역시"],
        ["Daegu", "Daegu City", "Daegu Metropolitan City"],
        ["大邱", "大邱市", "大邱广域市"],
        ["テグ", "テグ市", "テグ広域市"]
    ],
    "광주": [
        ["광주", "광주시", "광주광역시"],
        ["Gwangju", "Gwangju City", "Gwangju Metropolitan City"],
        ["光州", "光州市", "光州广域市"],
        ["クァンジュ", "クァンジュ市", "クァンジュ広域市"]
    ],
    "대전": [
        ["대전", "대전시", "대전광역시"],
        ["Daejeon", "Daejeon City", "Daejeon Metropolitan City"],
        ["大田", "大田市", "大田广域市"],
        ["テジョン", "テジョン市", "テジョン広域市"]
    ],
    "울산": [
        ["울산", "울산시", "울산광역시"],
        ["Ulsan", "Ulsan City", "Ulsan Metropolitan City"],
        ["蔚山", "蔚山市", "蔚山广域市"],
        ["ウルサン", "ウルサン市", "ウルサン広域市"]
    ],
    "세종": [
        ["세종", "세종시", "세종특별자치시"],
        ["Sejong", "Sejong City", "Sejong Special Self-Governing City"],
        ["世宗", "世宗市", "世宗特别自治市"],
        ["セジョン", "セジョン市", "セジョン特別自治市"]
    ],
    "강원": [
        ["강원", "강원도"],
        ["Gangwon", "Gangwon Province"],
        ["江原", "江原道"],
        ["カンウォン", "カンウォン道"]
    ],
    "충북": [
        ["충북", "충청북도"],
        ["Chungbuk", "North Chungcheong Province"],
        ["忠北", "忠清北道"],
        ["チュンブク", "忠清北道"]
    ],
    "충남": [
        ["충남", "충청남도"],
        ["Chungnam", "South Chungcheong Province"],
        ["忠南", "忠清南道"],
        ["チュンナム", "忠清南道"]
    ],
    "전북": [
        ["전북", "전라북도"],
        ["Jeonbuk", "North Jeolla Province"],
        ["全北", "全罗北道"],
        ["チョンブク", "全羅北道"]
    ],
    "전남": [
        ["전남", "전라남도"],
        ["Jeonnam", "South Jeolla Province"],
        ["全南", "全罗南道"],
        ["チョンナム", "全羅南道"]
    ],
    "경북": [
        ["경북", "경상북도"],
        ["Gyeongbuk", "North Gyeongsang Province"],
        ["庆北", "庆尚北道"],
        ["キョンブク", "慶尚北道"]
    ],
    "경남": [
        ["경남", "경상남도"],
        ["Gyeongnam", "South Gyeongsang Province"],
        ["庆南", "庆尚南道"],
        ["キョンナム", "慶尚南道"]
    ],
    "제주": [
        ["제주", "제주도", "제주특별자치도"],
        ["Jeju", "Jeju Island", "Jeju Special Self-Governing Province"],
        ["济州", "济州岛", "济州特别自治道"],
        ["チェジュ", "済州島", "済州特別自治道"]
    ]
}

# 지역별 대학교 정보 (예시, 실제 데이터로 채워넣어야 함)
university_by_region = {
    "서울": [
        ["서울대학교", "Seoul National University", "ソウル大学校", "首尔大学"],
        ["고려대학교", "Korea University", "高麗大学校", "高丽大学"],
        ["연세대학교", "Yonsei University", "延世大学校", "延世大学"],
        ["서강대학교", "Sogang University", "西江大学校", "西江大学"],
        ["성균관대학교", "Sungkyunkwan University", "成均館大学校", "成均馆大学"],
        ["한양대학교", "Hanyang University", "漢陽大学校", "汉阳大学"],
        ["중앙대학교", "Chung-Ang University", "中央大学校", "中央大学"],
        ["경희대학교", "Kyung Hee University", "慶熙大学校", "庆熙大学"],
        ["홍익대학교", "Hongik University", "弘益大学校", "弘益大学"],
        ["동국대학교", "Dongguk University", "東国大学校", "东国大学"],
        ["건국대학교", "Konkuk University", "建国大学校", "建国大学"],
        ["숙명여자대학교", "Sookmyung Women's University", "淑明女子大学校", "淑明女子大学"],
        ["이화여자대학교", "Ewha Womans University", "梨花女子大学校", "梨花女子大学"],
        ["한국외국어대학교", "Hankuk University of Foreign Studies", "韓国外国語大学校", "韩国外国语大学"],
        ["서울시립대학교", "University of Seoul", "ソウル市立大学校", "首尔市立大学"],
        ["숭실대학교", "Soongsil University", "崇實大学校", "崇实大学"],
        ["세종대학교", "Sejong University", "世宗大学校", "世宗大学"],
        ["국민대학교", "Kookmin University", "国民大学校", "国民大学"],
        ["덕성여자대학교", "Duksung Women's University", "徳成女子大学校", "德成女子大学"],
        ["동덕여자대학교", "Dongduk Women's University", "同德女子大学校", "同德女子大学"],
        ["서울과학기술대학교", "Seoul National University of Science and Technology", "ソウル科学技術大学校", "首尔科学技术大学"],
        ["삼육대학교", "Sahmyook University", "三育大学校", "三育大学"],
        ["상명대학교", "Sangmyung University", "相明大学校", "相明大学"],
        ["성신여자대학교", "Sungshin Women's University", "誠信女子大学校", "诚信女子大学"],
        ["한성대학교", "Hansung University", "韓成大学校", "韩成大学"],
        ["KC대학교", "KC University", "KC大学校", "KC大学"],
        ["감리교신학대학교", "Methodist Theological University", "監理教神学大学校", "监理教神学大学"],
        ["서울기독대학교", "Seoul Christian University", "ソウルキリスト教大学校", "首尔基督教大学"],
        ["서울장신대학교", "Seoul Jangsin University", "ソウル長神大学校", "首尔长神大学"],
        ["성공회대학교", "Sungkonghoe University", "聖公会大学校", "圣公会大学"],
        ["총신대학교", "Chongshin University", "総神大学校", "总神大学"],
        ["추계예술대학교", "Chugye University for the Arts", "秋渓芸術大学校", "秋溪艺术大学"],
        ["한국성서대학교", "Korean Bible University", "韓国聖書大学校", "韩国圣经大学"],
        ["한국체육대학교", "Korea National Sport University", "韓国体育大学校", "韩国体育大学"],
        ["한영신학대학교", "Hanying Theological University", "韓英神学大学校", "韩英神学大学"]
    ],
    "경기": [
        ["아주대학교", "Ajou University", "亜州大学校", "亚洲大学"],
        ["성균관대학교(자연과학캠퍼스)", "Sungkyunkwan University (Natural Sciences Campus)", "成均館大学校（自然科学キャンパス）", "成均馆大学（自然科学校区）"],
        ["한국외국어대학교(글로벌캠퍼스)", "Hankuk University of Foreign Studies (Global Campus)", "韓国外国語大学校（グローバルキャンパス）", "韩国外国语大学（国际校区）"],
        ["경희대학교(국제캠퍼스)", "Kyung Hee University (Global Campus)", "慶熙大学校（国際キャンパス）", "庆熙大学（国际校区）"],
        ["가천대학교", "Gachon University", "加川大学校", "加川大学"],
        ["경기대학교", "Kyonggi University", "京畿大学校", "京畿大学"],
        ["단국대학교", "Dankook University", "檀国大学校", "檀国大学"],
        ["한양대학교(ERICA)", "Hanyang University (ERICA Campus)", "漢陽大学校（ERICA）", "汉阳大学（ERICA校区）"],
        ["명지대학교", "Myongji University", "明知大学校", "明知大学"],
        ["강남대학교", "Kangnam University", "江南大学校", "江南大学"],
        ["경동대학교", "Kyungdong University", "京東大学校", "京东大学"],
        ["수원대학교", "University of Suwon", "水原大学校", "水原大学"],
        ["신한대학교", "Shinhan University", "新韓大学校", "新韩大学"],
        ["안양대학교", "Anyang University", "安養大学校", "安养大学"],
        ["용인대학교", "Yongin University", "龍仁大学校", "龙仁大学"],
        ["을지대학교", "Eulji University", "乙支大学校", "乙支大学"],
        ["평택대학교", "Pyeongtaek University", "平澤大学校", "平泽大学"],
        ["한경대학교", "Hankyong National University", "韓京大学校", "韩京大学"],
        ["한국산업기술대학교", "Korea Polytechnic University", "韓国産業技術大学校", "韩国产业技术大学"],
        ["한국항공대학교", "Korea Aerospace University", "韓国航空大学校", "韩国航空大学"],
        ["한세대학교", "Hansei University", "韓世大学校", "韩世大学"],
        ["협성대학교", "Hyupsung University", "協成大学校", "协成大学"],
        ["가톨릭대학교", "The Catholic University of Korea", "カトリック大学校", "天主教大学"],
        ["루터대학교", "Luther University", "ルーテル大学校", "路德大学"],
        ["서울신학대학교", "Seoul Theological University", "ソウル神学大学校", "首尔神学大学"],
        ["성결대학교", "Sungkyul University", "聖潔大学校", "圣洁大学"],
        ["중앙승가대학교", "Joong-Ang Sangha University", "中央僧伽大学校", "中央僧伽大学"],
        ["칼빈대학교", "Calvin University", "カルバン大学校", "加尔文大学"]
    ],
    "인천": [
        ["인천대학교", "Incheon National University", "仁川大学校", "仁川大学"],
        ["인하대학교", "Inha University", "仁荷大学校", "仁荷大学"],
        ["가천대학교(메디컬캠퍼스)", "Gachon University (Medical Campus)", "加川大学校（メディカルキャンパス）", "加川大学（医学校区）"],
        ["경인교육대학교", "Gyeongin National University of Education", "京仁教育大学校", "京仁教育大学"],
        ["인천가톨릭대학교", "Incheon Catholic University", "仁川カトリック大学校", "仁川天主教大学"]
    ],
    "부산": [
        ["부산대학교", "Pusan National University", "釜山大学校", "釜山大学"],
        ["동아대학교", "Dong-A University", "東亜大学校", "东亚大学"],
        ["부경대학교", "Pukyong National University", "釜慶大学校", "釜庆大学"],
        ["동의대학교", "Dong-Eui University", "東義大学校", "东义大学"],
        ["경성대학교", "Kyungsung University", "慶星大学校", "庆星大学"],
        ["신라대학교", "Silla University", "新羅大学校", "新罗大学"],
        ["고신대학교", "Kosin University", "高神大学校", "高神大学"],
        ["부산외국어대학교", "Busan University of Foreign Studies", "釜山外国語大学校", "釜山外国语大学"],
        ["동서대학교", "Dongseo University", "東西大学校", "东西大学"],
        ["한국해양대학교", "Korea Maritime and Ocean University", "韓国海洋大学校", "韩国海洋大学"],
        ["부산가톨릭대학교", "Catholic University of Pusan", "釜山カトリック大学校", "釜山天主教大学"]
    ],
    "대구": [
        ["경북대학교", "Kyungpook National University", "慶北大学校", "庆北大学"],
        ["계명대학교", "Keimyung University", "啓明大学校", "启明大学"],
        ["영남대학교", "Yeungnam University", "嶺南大学校", "岭南大学"],
        ["대구대학교", "Daegu University", "大邱大学校", "大邱大学"],
        ["대구가톨릭대학교", "Daegu Catholic University", "大邱カトリック大学校", "大邱天主教大学"],
        ["대구한의대학교", "Daegu Haany University", "大邱韓医大学校", "大邱韩医大学"],
        ["금오공과대학교", "Kumoh National Institute of Technology", "金烏工科大学校", "金乌工科大学"],
        ["경일대학교", "Kyungil University", "慶一大学校", "庆一大学"],
        ["대구예술대학교", "Daegu Arts University", "大邱芸術大学校", "大邱艺术大学"]
    ],
    "광주": [
        ["전남대학교", "Chonnam National University", "全南大学校", "全南大学"],
        ["조선대학교", "Chosun University", "朝鮮大学校", "朝鲜大学"],
        ["광주과학기술원", "Gwangju Institute of Science and Technology", "光州科学技術院", "光州科学技术院"],
        ["호남대학교", "Honam University", "湖南大学校", "湖南大学"],
        ["광주대학교", "Gwangju University", "光州大学校", "光州大学"],
        ["광주여자대학교", "Kwangju Women's University", "光州女子大学校", "光州女子大学"],
        ["남부대학교", "Nambu University", "南部大学校", "南部大学"],
        ["송원대학교", "Songwon University", "松原大学校", "松原大学"]
    ],
    "대전": [
        ["충남대학교", "Chungnam National University", "忠南大学校", "忠南大学"],
        ["한국과학기술원(KAIST)", "Korea Advanced Institute of Science and Technology (KAIST)", "韓国科学技術院（KAIST）", "韩国科学技术院（KAIST）"],
        ["한밭대학교", "Hanbat National University", "韓吧大学校", "韩吧大学"],
        ["대전대학교", "Daejeon University", "大田大学校", "大田大学"],
        ["배재대학교", "Pai Chai University", "培材大学校", "培材大学"],
        ["우송대학교", "Woosong University", "又松大学校", "又松大学"],
        ["을지대학교(대전캠퍼스)", "Eulji University (Daejeon Campus)", "乙支大学校（大田キャンパス）", "乙支大学（大田校区）"],
        ["침례신학대학교", "Korea Baptist Theological University", "浸礼神学大学校", "浸礼神学大学"],
        ["한남대학교", "Hannam University", "韓南大学校", "韩南大学"]
    ],
    "울산": [
        ["울산대학교", "University of Ulsan", "蔚山大学校", "蔚山大学"],
        ["울산과학기술원(UNIST)", "Ulsan National Institute of Science and Technology (UNIST)", "蔚山科学技術院（UNIST）", "蔚山科学技术院（UNIST）"],
        ["울산과학대학교", "Ulsan College", "蔚山科学大学校", "蔚山科学大学"]
    ],
    "세종": [
        ["고려대학교(세종캠퍼스)", "Korea University (Sejong Campus)", "高麗大学校（世宗キャンパス）", "高丽大学（世宗校区）"],
        ["홍익대학교(세종캠퍼스)", "Hongik University (Sejong Campus)", "弘益大学校（世宗キャンパス）", "弘益大学（世宗校区）"]
    ],
    "강원": [
        ["강원대학교", "Kangwon National University", "江原大学校", "江原大学"],
        ["연세대학교(미래캠퍼스)", "Yonsei University (Mirae Campus)", "延世大学校（未来キャンパス）", "延世大学（未来校区）"],
        ["강릉원주대학교", "Gangneung-Wonju National University", "江陵原州大学校", "江陵原州大学"],
        ["한림대학교", "Hallym University", "翰林大学校", "翰林大学"],
        ["춘천교육대학교", "Chuncheon National University of Education", "春川教育大学校", "春川教育大学"],
        ["강원도립대학교", "Gangwon Provincial College", "江原道立大学校", "江原道立大学"],
        ["상지대학교", "Sangji University", "尚志大学校", "尚志大学"],
        ["가톨릭관동대학교", "Catholic Kwandong University", "カトリック関東大学校", "天主教关东大学"],
        ["경동대학교", "Kyungdong University", "京東大学校", "京东大学"],
        ["한라대학교", "Halla University", "漢拏大学校", "汉拿大学"]
    ],
    "충북": [
        ["충북대학교", "Chungbuk National University", "忠北大学校", "忠北大学"],
        ["청주대학교", "Cheongju University", "清州大学校", "清州大学"],
        ["서원대학교", "Seowon University", "西原大学校", "西原大学"],
        ["세명대학교", "Semyung University", "世明大学校", "世明大学"],
        ["충주대학교", "Chungju National University", "忠州大学校", "忠州大学"],
        ["극동대학교", "Far East University", "極東大学校", "远东大学"],
        ["중원대학교", "Jungwon University", "中原大学校", "中原大学"],
        ["건국대학교(글로컬캠퍼스)", "Konkuk University (GLOCAL Campus)", "建国大学校（グローカルキャンパス）", "建国大学（全球本地化校区）"],
        ["한국교통대학교", "Korea National University of Transportation", "韓国交通大学校", "韩国交通大学"]
    ],
    "충남": [
        ["충남대학교", "Chungnam National University", "忠南大学校", "忠南大学"],
        ["공주대학교", "Kongju National University", "公州大学校", "公州大学"],
        ["순천향대학교", "Soonchunhyang University", "順天郷大学校", "顺天乡大学"],
        ["남서울대학교", "Namseoul University", "南ソウル大学校", "南首尔大学"],
        ["건양대학교", "Konyang University", "建陽大学校", "建阳大学"],
        ["백석대학교", "Baekseok University", "白石大学校", "白石大学"],
        ["호서대학교", "Hoseo University", "湖西大学校", "湖西大学"],
        ["선문대학교", "Sun Moon University", "鮮文大学校", "鲜文大学"],
        ["한서대학교", "Hanseo University", "韓瑞大学校", "韩瑞大学"],
        ["나사렛대학교", "Korea Nazarene University", "ナザレ大学校", "拿撒勒大学"],
        ["중부대학교", "Joongbu University", "中部大学校", "中部大学"],
        ["청운대학교", "Chungwoon University", "清雲大学校", "清云大学"]
    ],
    "전북": [
        ["전북대학교", "Jeonbuk National University", "全北大学校", "全北大学"],
        ["전주대학교", "Jeonju University", "全州大学校", "全州大学"],
        ["원광대학교", "Wonkwang University", "圓光大学校", "圆光大学"],
        ["군산대학교", "Kunsan National University", "群山大学校", "群山大学"],
        ["우석대학교", "Woosuk University", "又石大学校", "又石大学"],
        ["예수대학교", "Jesus University", "イエス大学校", "耶稣大学"],
        ["한일장신대학교", "Hanil University and Presbyterian Theological Seminary", "韓一長神大学校", "韩一长神大学"],
        ["호원대학교", "Howon University", "湖原大学校", "湖原大学"]
    ],
    "전남": [
        ["전남대학교(여수캠퍼스)", "Chonnam National University (Yeosu Campus)", "全南大学校（麗水キャンパス）", "全南大学（丽水校区）"],
        ["순천대학교", "Sunchon National University", "順天大学校", "顺天大学"],
        ["목포대학교", "Mokpo National University", "木浦大学校", "木浦大学"],
        ["동신대학교", "Dongshin University", "東新大学校", "东新大学"],
        ["세한대학교", "Sehan University", "世韓大学校", "世韩大学"],
        ["초당대학교", "Chodang University", "草堂大学校", "草堂大学"],
        ["목포해양대학교", "Mokpo National Maritime University", "木浦海洋大学校", "木浦海洋大学"]
    ],
    "경북": [
        ["경북대학교(상주캠퍼스)", "Kyungpook National University (Sangju Campus)", "慶北大学校（尚州キャンパス）", "庆北大学（尚州校区）"],
        ["포항공과대학교(POSTECH)", "Pohang University of Science and Technology (POSTECH)", "浦項工科大学校（POSTECH）", "浦项工科大学（POSTECH）"],
        ["안동대학교", "Andong National University", "安東大学校", "安东大学"],
        ["경주대학교", "Gyeongju University", "慶州大学校", "庆州大学"],
        ["김천대학교", "Gimcheon University", "金泉大学校", "金泉大学"],
        ["대구가톨릭대학교(경산캠퍼스)", "Daegu Catholic University (Gyeongsan Campus)", "大邱カトリック大学校（慶山キャンパス）", "大邱天主教大学（庆山校区）"],
        ["동국대학교(경주캠퍼스)", "Dongguk University (Gyeongju Campus)", "東国大学校（慶州キャンパス）", "东国大学（庆州校区）"],
        ["위덕대학교", "Uiduk University", "威德大学校", "威德大学"],
        ["한동대학교", "Handong Global University", "韓東大学校", "韩东大学"]
    ],
    "경남": [
        ["경상국립대학교", "Gyeongsang National University", "慶尚国立大学校", "庆尚国立大学"],
        ["창원대학교", "Changwon National University", "昌原大学校", "昌原大学"],
        ["인제대학교", "Inje University", "仁済大学校", "仁济大学"],
        ["경남대학교", "Kyungnam University", "慶南大学校", "庆南大学"],
        ["영산대학교", "Youngsan University", "霊山大学校", "灵山大学"],
        ["울산대학교", "University of Ulsan", "蔚山大学校", "蔚山大学"],
        ["한국해양대학교(통영캠퍼스)", "Korea Maritime and Ocean University (Tongyeong Campus)", "韓国海洋大学校（統営キャンパス）", "韩国海洋大学（统营校区）"],
        ["진주교육대학교", "Chinju National University of Education", "晋州教育大学校", "晋州教育大学"]
    ],
    "제주": [
        ["제주대학교", "Jeju National University", "済州大学校", "济州大学"],
        ["제주국제대학교", "Jeju International University", "済州国際大学校", "济州国际大学"],
        ["탐라대학교", "Tamna University", "耽羅大学校", "耽罗大学"]
    ]
}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return session_memories[session_id]

def extract_regions(text: str) -> List[str]:
    nouns = kkma.nouns(text)
    extracted_regions = []
    for noun in nouns:
        for region, aliases in region_dict.items():
            if noun in aliases[0]:
                extracted_regions.append(region)
                break
    return extracted_regions

def get_universities_by_region(regions: List[str]) -> Dict[str, List[str]]:
    result = {}
    for region in regions:
        if region in university_by_region:
            result[region] = university_by_region[region]
    return result

def preprocess_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    regions = extract_regions(text)
    for region in regions:
        text = text.replace(region, f"[REGION]{region}[/REGION]")
    
    return text.strip()

def process_pdf_pages(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    processed_pages = []
    for page in pages:
        content = preprocess_text(page.page_content)
        regions = extract_regions(content)
        splits = text_splitter.split_text(content)
        
        page_chunks = []
        for split in splits:
            summary = summarize_text(split)
            doc = Document(page_content=summary, metadata={"page": page.metadata["page"], "regions": regions})
            page_chunks.append(doc)
        
        processed_pages.append(page_chunks)
    
    return processed_pages


def process_and_save_data():
    global es_client

    index_name = "pdf_search"
    
    mappings = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1536  # OpenAI의 text-embedding-ada-002 모델 사용 시
                },
                "metadata": {
                    "properties": {
                        "page": {"type": "integer"},
                        "regions": {"type": "keyword"}
                    }
                }
            }
        }
    }
    
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mappings)
    
    processed_pages = process_pdf_pages(pdf_path)
    
    for page_chunks in processed_pages:
        actions = []
        for doc in page_chunks:
            embedding = embeddings.embed_query(doc.page_content)
            actions.append({
                "_op_type": "index",
                "_index": index_name,
                "_source": {
                    "content": doc.page_content,
                    "content_vector": embedding,
                    "metadata": doc.metadata
                }
            })
        if actions:
            helpers.bulk(es_client, actions)
    
    es_client.indices.refresh(index=index_name)

def hybrid_search(query: str, top_k: int = 5):
    query_vector = embeddings.embed_query(query)
    
    script_score = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    hybrid_query = {
        "bool": {
            "should": [
                {"match": {"content": query}},
                script_score
            ]
        }
    }
    
    try:
        response = es_client.search(
            index="pdf_search",
            body={
                "query": hybrid_query,
                "size": top_k
            }
        )
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Elasticsearch 검색 중 오류 발생: {str(e)}")
        # 오류 발생 시 빈 리스트 반환
        return []

def initialize_langchain():
    global agent_executor, es_client, es_retriever
    
    es_retriever = ElasticsearchRetriever(
        client=es_client,
        index_name="pdf_search",
        k=5
    )
    
    retriever_tool = create_retriever_tool(
        es_retriever,
        "pdf_search",
        "이 도구를 사용하여 PDF 파일에서 외국인 학생 입학 계획에 대한 정보를 검색할 수 있습니다. 또한 언급된 지역의 대학교 정보도 제공합니다."
    )

    tools = [retriever_tool]

    openai = ChatOpenAI(
        model="gpt-3.5-turbo", 
        api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0.1,
        max_tokens=1000
    )

    prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    """
    당신은 한국 대학의 2025학년도 재외국민과 외국인 특별전형 전문가입니다. 제공된 PDF 문서, university_by_region 데이터, region_dict 데이터를 바탕으로 간결하고 정확하게 답변하세요.
    주요 지침:

    사용자 질문을 정확히 파악하고 요점만 답변하세요.
    질문자가 매번 질문을 할 때마다 university_by_region과 region_dict를 반드시 확인하고 답변하세요.
    대학교 목록이나 모집단위를 요청받으면 모든 대학교와 모집단위를 나열하세요. 단, 모집단위가 너무 많으면 일부만 보여주고 "등"을 붙이세요. 대학교명을 요청받으면 모든 대학교명을 생략하지 말고 다 알려주세요.
    숫자 정보(대학 수, 모집 인원 등)를 물으면 정확한 숫자만 답변하세요.
    특정 대학의 전공이나 모집 인원 정보는 PDF 문서의 21페이지부터 166페이지를 참조하여 정확히 제공하세요.
    대학별 모집유형 정보는 PDF의 11페이지에서 20페이지를 참고하세요.
    대학입학전형기본사항은 PDF의 1페이지부터 10페이지를 참고하세요.
    불필요한 설명이나 부가 정보는 제공하지 마세요.
    정보가 없으면 "해당 정보를 찾을 수 없습니다."라고만 답변하세요.
    외국인 전형 정보를 물어보면 반드시 PDF의 '전형방법 및 전형요소' 섹션(167페이지 이후)을 참고하여 답변하세요.
    중복된 정보는 생략하고 한 번만 제공하세요.

    답변 예시:
    User: 경북에 있는 대학교들을 알려줘
    Assistant: 경북대학교(상주캠퍼스), 포항공과대학교(POSTECH), 안동대학교, 경주대학교, 김천대학교, 대구가톨릭대학교(경산캠퍼스), 동국대학교(경주캠퍼스), 위덕대학교, 한동대학교입니다.
    User: 경북에 있는 대학교 수는?
    Assistant: 9개입니다.
    User: 위덕대학교에 외국인 전형이 있는 전공을 알려줘
    Assistant: 불교문화학과, 한국어학부, 일본언어문화학과, 경찰정보보안학과, 경영학과, 사회복지학과, 항공호텔서비스학과, 유아교육과, 외식조리제과제빵학부, 지능형전력시스템공학과, 건강스포츠학부입니다.
    User: 군산대학교(전북)의 외국인 전형 전공들의 모집 인원을 알려줘
    Assistant: 33명입니다.
    User: 서울대학교의 외국인 전형 방법을 알려줘
    Assistant: 서류평가 100%입니다.
    User: 서강대학교의 외국인 전형이 있는지 알려줘
    Assistant: 있습니다. 모집단위로는 국제인문학부, 사회과학부, 경제학부, 경영학부, 자연과학부, 공학부, 컴퓨터공학과, 전자공학과, 국어국문학과, 영미어문, 유럽문화, 중국문화 등이 있습니다.
    User: 경남에 있는 대학교들을 알려줘
    Assistant: 경상국립대학교, 창원대학교, 경남대학교, 인제대학교입니다.
    User: 경상국립대학교의 외국인 전형 전공을 알려줘
    Assistant: 국어국문학과, 영어영문학과, 독일학과, 러시아학과, 중어중문학과, 사학과, 철학과, 불어불문학과, 일어일문학과, 민속무용학과, 한문학과, 법학과, 행정학과, 정치외교학과, 사회학과, 경제학과, 경영학부, 회계학과, 국제통상학과, 심리학과, 사회복지학과, 아동가족학과, 시각디자인학과 등이 있습니다.
    """
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    extract_prompt = PromptTemplate(
        input_variables=["query"],
        template="다음 질문에서 중요한 키워드와 의도를 추출하세요: {query}"
    )

    response_prompt = PromptTemplate(
        input_variables=["extracted_info", "agent_response"],
        template="추출된 정보: {extracted_info}\n\n에이전트 응답: {agent_response}\n\n위 정보를 바탕으로 최종 응답을 생성하세요."
    )

    extract_chain = LLMChain(llm=openai, prompt=extract_prompt, output_key="extracted_info")
    response_chain = LLMChain(llm=openai, prompt=response_prompt, output_key="final_response")

    overall_chain = SequentialChain(
        chains=[extract_chain, response_chain],
        input_variables=["query", "agent_response"],
        output_variables=["extracted_info", "final_response"],
        verbose=True
    )

    print("Agent executor and pipeline initialized")
    return agent_executor, overall_chain

def summarize_text(text: str, max_tokens: int = 200) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"주어진 텍스트를 {max_tokens}단어 이내로 간결하게 요약하십시오. 주요 포인트만 포함하도록 합니다."),
        ("human", "{input}")
    ])
    
    openai = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1,
        max_tokens=max_tokens
    )
    
    response = openai(prompt.format_prompt(input=text).to_messages())
    return response.content

def manage_chat_history(memory: ConversationBufferMemory, max_messages: int = 5, max_tokens: int = 1000):
    messages = memory.chat_memory.messages
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    
    total_tokens = sum(len(m.content.split()) for m in messages)
    while total_tokens > max_tokens and len(messages) > 2:
        removed = messages.pop(0)
        total_tokens -= len(removed.content.split())
    
    memory.chat_memory.messages = messages

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, es_client
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    if not elasticsearch_url.startswith(('http://', 'https://')):
        elasticsearch_url = f"http://{elasticsearch_url}"
    
    print(f"Connecting to Elasticsearch at: {elasticsearch_url}")
    
    try:
        es_client = Elasticsearch([elasticsearch_url])
        
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        
        print("Successfully connected to Elasticsearch")
        
        if not es_client.indices.exists(index="pdf_search"):
            process_and_save_data()
       
        initialize_langchain()
        yield
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print(traceback.format_exc())
        raise
    finally:
        if es_client:
            es_client.close()

app = FastAPI(lifespan=lifespan)
from langchain.schema import AIMessage
import traceback
from collections import defaultdict
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/study")
def query(query: Query, background_tasks: BackgroundTasks):
    global agent_executor
    if agent_executor is None:
        return {"message": "서버 초기화 중입니다. 잠시 후 다시 시도해 주세요."}
    
    try:
        memory = get_memory(query.session_id)
        
        regions = extract_regions(query.input)
        universities = get_universities_by_region(regions)
        
        university_info = "\n".join([f"{region}: {', '.join([uni[0] for uni in unis[:2]])}" for region, unis in universities.items()])
        short_query = query.input[:200]
        enhanced_query = f"{short_query}\n지역: {', '.join(regions)}\n대학교: {', '.join([uni[0] for unis in universities.values() for uni in unis[:1]])}"
        
        manage_chat_history(memory, max_messages=2, max_tokens=300)
        
        search_results = hybrid_search(query.input, top_k=10)
    
        filtered_results = []

        if search_results:  # 검색 결과가 있을 경우에만 처리
            for hit in search_results:
                content = hit["_source"]["content"]
                hit_regions = extract_regions(content)
                if any(region in regions for region in hit_regions):
                    filtered_results.append(content)
            
            if filtered_results:
                result_text = "\n".join(filtered_results[:5])  # 상위 5개 결과만 사용
                summarized_result = summarize_text(result_text, max_tokens=100)
            else:
                summarized_result = "관련 정보를 찾을 수 없습니다."
        else:
            summarized_result = "검색 결과가 없습니다."
    
        agent_input = f"{enhanced_query[:100]}\n요약: {summarized_result[:200]}"
        
        agent_executor, overall_chain = initialize_langchain()
        
        agent_response = agent_executor.invoke({
            "input": agent_input, 
            "chat_history": memory.chat_memory.messages
        })
        
        if isinstance(agent_response, dict) and "output" in agent_response:
            agent_output = agent_response["output"]
        elif isinstance(agent_response, AIMessage):
            agent_output = agent_response.content
        else:
            agent_output = str(agent_response)
        
        # 파이프라인 실행
        pipeline_response = overall_chain({
            "query": query.input,
            "agent_response": agent_output
        })
        
        final_response = pipeline_response["final_response"]
        
        # 응답 길이 제한
        final_response = final_response[:500]
        
        final_response_with_info = f"{final_response}\n\n추가 대학교 정보:\n{university_info[:200]}"
        
        memory.chat_memory.add_user_message(query.input[:100])
        memory.chat_memory.add_ai_message(final_response_with_info[:200])
        
        return {"response": final_response_with_info, "extracted_regions": regions, "related_universities": universities}
    except Exception as e:
        print(f"쿼리 함수에서 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")