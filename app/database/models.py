from sqlalchemy import Column, Integer, String, Text, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base
from datetime import date

class File(Base):
    __tablename__ = "TFILE"

    FILE_NO = Column(Integer, primary_key=True, autoincrement=True)
    CATEGORY_NO = Column(Integer, ForeignKey("TCATEGORY.CATEGORY_NO"), nullable=False)
    FILE_ORIGIN_NAME = Column(String(255), nullable=False)
    FILE_SAVE_NAME = Column(String(255), nullable=False)
    FILE_PATH = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    category = relationship("Category", back_populates="files")
    insert_user = relationship("User", foreign_keys=[INSERT_USER_NO], back_populates="inserted_files")
    modify_user = relationship("User", foreign_keys=[MODIFY_USER_NO], back_populates="modified_files")
    user = relationship("User", back_populates="file", foreign_keys="[User.FILE_NO]")

class User(Base):
    __tablename__ = "TUSER"

    USER_NO = Column(Integer, primary_key=True, autoincrement=True)
    USER_NAME = Column(String(255), nullable=False)
    GRADE = Column(String(255), nullable=False)
    EMAIL = Column(String(255), nullable=False, unique=True)
    PASSWORD = Column(String(255), nullable=False)
    NATIONLITY = Column(String(255), nullable=False)
    USER_HP = Column(String(255), nullable=False)
    FILE_NO = Column(Integer, ForeignKey("TFILE.FILE_NO"))
    ACTIVE_YN = Column(Boolean, default=True)
    WITHDRAW_YN = Column(Boolean, default=False)
    ANONYMIZE_YN = Column(Boolean, default=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    file = relationship("File", foreign_keys=[FILE_NO], back_populates="user")
    inserted_files = relationship("File", foreign_keys=[File.INSERT_USER_NO], back_populates="insert_user")
    modified_files = relationship("File", foreign_keys=[File.MODIFY_USER_NO], back_populates="modify_user")

class CategoryGb(Base):
    __tablename__ = "TCATEGORYGB"

    CATEGORY_GB_NO = Column(Integer, primary_key=True, autoincrement=True)
    CATEGORY_GB_NAME = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    categories = relationship("Category", back_populates="category_gb")

class Category(Base):
    __tablename__ = "TCATEGORY"

    CATEGORY_NO = Column(Integer, primary_key=True, autoincrement=True)
    CATEGORY_GB_NO = Column(Integer, ForeignKey("TCATEGORYGB.CATEGORY_GB_NO"), nullable=False)
    CATEGORY_NAME = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    category_gb = relationship("CategoryGb", back_populates="categories")
    files = relationship("File", back_populates="category")
    boards = relationship("Board", back_populates="category")
    chat_histories = relationship("ChatHis", back_populates="category")

class Board(Base):
    __tablename__ = "TBOARD"

    BOARD_NO = Column(Integer, primary_key=True, autoincrement=True)
    CATEGORY_NO = Column(Integer, ForeignKey("TCATEGORY.CATEGORY_NO"), nullable=False)
    TITLE = Column(String(255), nullable=False)
    CONTENT = Column(String(255), nullable=False)
    SUMMARY = Column(String(255))
    VIEWS = Column(Integer, nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    category = relationship("Category", back_populates="boards")
    comments = relationship("BoardComment", back_populates="board")
    files = relationship("BoardFile", back_populates="board")

class BoardComment(Base):
    __tablename__ = "TBOARDCOMMENT"

    BOARD_NO = Column(Integer, ForeignKey("TBOARD.BOARD_NO"), primary_key=True)
    COMMENT_NO = Column(Integer, ForeignKey("TCOMMENT.COMMENT_NO"), primary_key=True)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    board = relationship("Board", back_populates="comments")
    comment = relationship("Comment", back_populates="board_comments")

class BoardFile(Base):
    __tablename__ = "TBOARDFILE"

    BOARD_NO = Column(Integer, ForeignKey("TBOARD.BOARD_NO"), primary_key=True)
    FILE_NO = Column(Integer, ForeignKey("TFILE.FILE_NO"), primary_key=True)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)

    board = relationship("Board", back_populates="files")
    file = relationship("File")

class ChatHis(Base):
    __tablename__ = "TCHATHIS"

    CHAT_HIS_NO = Column(Integer, primary_key=True, autoincrement=True)
    CHAT_HIS_SEQ = Column(Integer, primary_key=True)
    CATEGORY_NO = Column(Integer, ForeignKey("TCATEGORY.CATEGORY_NO"), nullable=False)
    QUESTION = Column(Text)
    ANSWER = Column(Text)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    category = relationship("Category", back_populates="chat_histories")

class Comment(Base):
    __tablename__ = "TCOMMENT"

    COMMENT_NO = Column(Integer, primary_key=True, autoincrement=True)
    CONTENT = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

    board_comments = relationship("BoardComment", back_populates="comment")

class Game(Base):
    __tablename__ = "TGAME"

    GAME_NO = Column(Integer, primary_key=True, autoincrement=True)
    SCORE = Column(Integer, nullable=False)
    INITIAL_CONSONANT = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)

class UserData(Base):
    __tablename__ = "TUSERDATA"

    USER_DATA_NO = Column(Integer, primary_key=True, autoincrement=True)
    DATA_FIELD = Column(String(255), nullable=False)
    DATA_VALUE = Column(String(255), nullable=False)
    INSERT_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    INSERT_DATE = Column(Date, nullable=False, default=date.today)
    MODIFY_USER_NO = Column(Integer, ForeignKey("TUSER.USER_NO"))
    MODIFY_DATE = Column(Date, nullable=False, default=date.today, onupdate=date.today)