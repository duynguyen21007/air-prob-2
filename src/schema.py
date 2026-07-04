from pydantic import BaseModel
from typing import List
from enum import Enum

class EntityStage1(BaseModel):
    text: str
    position: List[int]

class NERResponseStage1(BaseModel):
    entities: List[EntityStage1]


class EntityType(str, Enum):
    TRIEU_CHUNG = "TRIỆU_CHỨNG"
    THUOC = "THUỐC"
    CHAN_DOAN = "CHẨN_ĐOÁN"
    TEN_XET_NGHIEM = "TÊN_XÉT_NGHIỆM"
    KET_QUA_XET_NGHIEM = "KẾT_QUẢ_XÉT_NGHIỆM"

class EntityStage2(BaseModel):
    text: str
    position: List[int]
    type: EntityType

class ClassifyResponseStage2(BaseModel):
    entities: List[EntityStage2]


class EntityStage3(BaseModel):
    text: str
    position: List[int]
    type: EntityType
    assertions: List[str]

class AssertionsResponseStage3(BaseModel):
    entities: List[EntityStage3]
