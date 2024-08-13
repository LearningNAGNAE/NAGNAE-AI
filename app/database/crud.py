from sqlalchemy.orm import Session
from sqlalchemy import func
from . import models
from datetime import date

from sqlalchemy import func
from datetime import date

def create_chat_history(db: Session, user_no: int, category_no: int, question: str, answer: str, is_new_session: bool):
    try:
        if is_new_session:
            # 새 세션 시작: 새로운 CHAT_HIS_NO 생성 (auto-increment)
            chat_history = models.ChatHis(
                CATEGORY_NO=category_no,
                CHAT_HIS_SEQ=1,  # 새 세션은 항상 1부터 시작
                QUESTION=question,
                ANSWER=answer,
                INSERT_USER_NO=user_no,
                INSERT_DATE=date.today(),
                MODIFY_USER_NO=user_no,
                MODIFY_DATE=date.today()
            )
        else:
            # 기존 세션 계속: 가장 최근의 CHAT_HIS_NO 찾기
            latest_chat = db.query(models.ChatHis).filter(
                models.ChatHis.INSERT_USER_NO == user_no
            ).order_by(models.ChatHis.CHAT_HIS_NO.desc()).first()

            if latest_chat:
                new_chat_his_no = latest_chat.CHAT_HIS_NO
                new_chat_his_seq = db.query(func.max(models.ChatHis.CHAT_HIS_SEQ)).filter(
                    models.ChatHis.INSERT_USER_NO == user_no,
                    models.ChatHis.CHAT_HIS_NO == new_chat_his_no
                ).scalar() or 0
                new_chat_his_seq += 1
            else:
                # 사용자의 첫 채팅인 경우
                new_chat_his_no = None  # auto-increment가 처리할 것임
                new_chat_his_seq = 1

            chat_history = models.ChatHis(
                CHAT_HIS_NO=new_chat_his_no,  # None이면 auto-increment
                CHAT_HIS_SEQ=new_chat_his_seq,
                CATEGORY_NO=category_no,
                QUESTION=question,
                ANSWER=answer,
                INSERT_USER_NO=user_no,
                INSERT_DATE=date.today(),
                MODIFY_USER_NO=user_no,
                MODIFY_DATE=date.today()
            )

        db.add(chat_history)
        db.commit()
        db.refresh(chat_history)
        return chat_history
    except Exception as e:
        db.rollback()
        raise e