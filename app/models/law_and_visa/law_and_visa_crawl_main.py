from app.models.law_and_visa.law_and_visa_util import setup_driver
from law_and_visa_crawl import crawl_law_a, crawl_law_b

def main():
    crawl_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"
    crawl_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"
    
    driver = setup_driver()
    
    try:
        # print("종합법률정보, 판례")
        # crawl_law_a(driver, crawl_law_a_url)
        
        print("종합법률정보, 법률")
        crawl_law_b(driver, crawl_law_b_url)
        
    finally:
        driver.quit()

    print("완료")

if __name__ == "__main__":
    main()