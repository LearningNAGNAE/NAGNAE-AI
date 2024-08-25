from sqlalchemy.orm import Session
from app.database.models import Board
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from app.database.db import SessionLocal
import re
from bs4 import BeautifulSoup

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
scheduler = None

import re
from bs4 import BeautifulSoup
from langdetect import detect

from langdetect import detect
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(title: str, content: str) -> str:
    # 언어 감지
    try:
        lang = detect(title + " " + content)
    except:
        lang = 'en'  # 기본값으로 영어 설정
    
    # 짧은 내용 처리
    if len(content.strip()) < 10:
        short_messages = {
            'ko': "요약하기에는 너무 짧아요. 더 자세한 내용을 공유해주세요!",
            'en': "Too short to summarize. Please share more details!",
            'ja': "要約するには短すぎます。もう少し詳しく教えてください！",
            'zh': "内容太短，无法总结。请分享更多细节！",
            'vi': "Quá ngắn để tóm tắt. Vui lòng chia sẻ thêm chi tiết!",
            'th': "สั้นเกินไปที่จะสรุป โปรดแบ่งปันรายละเอียดเพิ่มเติม!",
        }
        return short_messages.get(lang, short_messages['en'])

    # 프롬프트 구성
    system_prompt = """
    # AI Assistant for Multilingual Community Board Summary

    ## Role and Responsibility
    You are a specialized AI assistant providing concise and culturally sensitive summaries for a community board used by foreigners living in Korea. Your primary goals are to:

    1. Provide accurate, concise, and easy-to-understand summaries in the language specified in the 'RESPONSE_LANGUAGE' field.
    2. Capture the essence of posts, highlighting cultural experiences, questions about life in Korea, and shared information among the foreign community.
    3. Ensure cultural sensitivity and awareness in all summaries.
    4. Foster understanding between different cultures and languages represented in the community.

    ## Guidelines

    1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the original post's language.

    2. Summary Scope:
       - Cultural Experiences: Highlight unique experiences or observations about life in Korea.
       - Questions and Advice: Summarize queries about living, working, or studying in Korea.
       - Information Sharing: Capture key points of shared information relevant to the foreign community.
       - Community Interaction: Note any calls for meetups, language exchanges, or community events.

    3. Specific Focus Areas:
       - Visa-related topics: Briefly mention if the post discusses visa issues without going into detail.
       - Cultural differences: Highlight any mentioned cultural contrasts or misunderstandings.
       - Language learning: Note any discussions about Korean language learning or language exchange.
       - Practical living tips: Summarize any advice on daily life in Korea (e.g., housing, transportation, healthcare).

    4. Completeness: Provide a concise summary in 2-3 sentences, capturing the main points of the post.

    5. Cultural Sensitivity: Be aware of and respect diverse cultural backgrounds. Avoid stereotypes or generalizations.

    6. Uncertainty Handling: If the post content is unclear or seems incomplete, mention this in your summary.

    7. Format: Provide the summary as plain text without any prefixes like "Summary:" or "글 요약:". Do not use any HTML tags or special formatting.

    Remember, your goal is to provide a brief, culturally sensitive summary that captures the essence of the post and fosters understanding within this diverse community of foreigners in Korea.
    """

    human_prompt = f"""
    RESPONSE_LANGUAGE: {lang}
    TITLE: {title}
    CONTENT: {content}

    Please provide a concise summary of the above post in the specified RESPONSE_LANGUAGE, capturing its main points and any cultural or community-related aspects. Your summary should be 2-3 sentences long. Do not include any prefixes, HTML tags, or special formatting in your response.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            max_tokens=200,
            n=1,
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error in summarize_text: {str(e)}")
        return ""
    

def update_summaries():
    db = SessionLocal()
    try:
        posts = db.query(Board).filter(Board.SUMMARY == None).all()
        for post in posts:
            summary = summarize_text(post.TITLE, post.CONTENT)
            if summary:
                post.SUMMARY = summary
                post.MODIFY_DATE = datetime.now()
                post.MODIFY_USER_NO = 1  # 시스템 사용자로 가정
        db.commit()
        print(f"Updated summaries for {len(posts)} posts")
    except Exception as e:
        db.rollback()
        print(f"Error updating summaries: {str(e)}")
    finally:
        db.close()

def start_scheduler():
    global scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_summaries, 'interval', minutes=30)
    scheduler.start()
    print("Scheduler started. Summaries will be updated every 30 minutes.")

def stop_scheduler():
    global scheduler
    if scheduler:
        scheduler.shutdown()
        print("Scheduler stopped.")

# 수동 업데이트를 위한 함수
async def manual_update_summaries():
    try:
        update_summaries()
        return {"message": "Summary update completed"}
    except Exception as e:
        print(f"Error in manual update: {str(e)}")
        return {"message": "Error occurred during summary update"}

# FastAPI의 shutdown 이벤트에 연결할 함수
async def shutdown_event():
    stop_scheduler()