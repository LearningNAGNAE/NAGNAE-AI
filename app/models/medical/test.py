import requests
from bs4 import BeautifulSoup

# 웹페이지 URL
url = "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=508&ccfNo=3&cciNo=6&cnpClsNo=1&search_put="

# 웹페이지 내용 가져오기
response = requests.get(url)

# BeautifulSoup 객체 생성
soup = BeautifulSoup(response.text, 'html.parser')

# <div class="ovDivbox"> 태그들 찾기
div_contents = soup.find_all('div', class_='ovDivbox')

if div_contents:
    # 파일로 저장
    with open('extracted_content.txt', 'w', encoding='utf-8') as file:
        for div in div_contents:
            # 태그 내의 텍스트 추출 및 정제
            text_content = div.get_text(strip=True, separator='\n')
            file.write(text_content + "\n\n")  # 각 div 내용 사이에 빈 줄 추가
    
    print("내용이 성공적으로 저장되었습니다.")
else:
    print("지정된 div 태그를 찾을 수 없습니다.")