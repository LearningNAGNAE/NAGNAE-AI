from pymilvus import connections

try:
    connections.connect("default", host="localhost", port="19530")
    print("Milvus 서버에 성공적으로 연결되었습니다.")
    connections.disconnect("default")
except Exception as e:
    print(f"Milvus 서버 연결 실패: {e}")