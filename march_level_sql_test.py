

# 어노테이션 파일 불러온 후 해당 난이도 맞는지 확인해보는 작업
# 필요정보는 추출하되 쿼리와 난이도를 중점
# 기존 난이도와 비교할수있는 설계 ex) qurey , 기존 : level , 신규 level 비교 동일 , 다름  extra hard는 -> extra로 변경 후 작업해볼것.
# 일치하지 않는다면 그것만 따로 추출하여 분석 리포팅이 과제
# labelon db에서 사용자 입력한 정보도 가져와서 동일하게 비교도 해야됨

from level_all import spider_level_query
import json


json_file = "C:/Users/yoonsub/Documents/database_nl2sql_220916/라벨링데이터/nl2sql_data_labeling.json"
with open(json_file , 'r' ,  encoding='utf8') as f:
    json_data = json.load(f)


#print(json_data['data'][0])
#print(len(json_data['data'])) # 10568개의 데이터


# {'db_id': 'seouldata_transportation_69', 'utterance_id': 'Hch_0001', 'hardness': 'easy',
# 'utterance_type': 'BR08', 'query': 'SELECT BUS_PSGR_CNT FROM TPSS_EMD_BUS',
# 'utterance': '행정동별 버스 승객수를 보여줘',
# 'values': [],
# 'cols': [{'token': '버스 승객수', 'start': 5, 'column_index': 3}]}

# test "query" , "utterance" , hardness

    error_count = 0
    real_count = 0
    not_match_count = 0
    count = 0


    easy_count = 0
    medium_count = 0
    hard_count = 0
    extra_hard_count = 0

    not_match_easy_count = 0
    not_match_medium_count = 0
    not_match_hard_count = 0
    not_match_extra_hard_count = 0

    not_match_csv = []
    not_match_list = {}
    not_match_list['data_not_match'] = []


for i in range(0, 10567) :
    while True :
        real_count +=1
        try:
            s = json_data['data'][i]
            level_human = s['hardness']
            query_human = s['query']

            if level_human == str(spider_level_query(query_human)[0]) : #
                count += 1
                if level_human == 'easy' :
                    easy_count += 1
                elif level_human == 'medium' :
                    medium_count +=1
                elif level_human == 'hard' :
                    hard_count +=1
                elif level_human == 'extra hard' :
                    extra_hard_count +=1
                #print( count , "일치 : " )

            else :

                if level_human != str(spider_level_query(query_human)[0]):  #
                    count += 1
                    if level_human == 'easy':
                        not_match_easy_count += 1
                    elif level_human == 'medium':
                        not_match_medium_count += 1
                    elif level_human == 'hard':
                        not_match_hard_count += 1
                    elif level_human == 'extra hard':
                        not_match_extra_hard_count += 1
                    # print( count , "일치 : " )
                not_match_count += 1


                not_match_list['data_not_match'].append({
                    "qurey" : query_human ,
                    "level_creator" : level_human ,
                    "level_spider" : str(spider_level_query(query_human)[0]),
                    "score" :  str(spider_level_query(query_human)[1])
                })

                print(count, "/" ,real_count,"불일치")
                print("쿼리문:", query_human)
                print("작성자 난이도 / SPIDER 난이도    : " , level_human , "/", str(spider_level_query(query_human)[0]))
                print("SPIDER 점수" ,str(spider_level_query(query_human)[1]))
                print("################################################################################################################")


        except Exception as e  :
            i = i + 1
            error_count += 1  # ???????? 왜 중간에 숫자가 뻥튀기 될까?
            continue
            #print(e ,"번호 : " , count , "/" , real_count , "예외가 발생하였습니다.")
        break



with open('C:/Users/yoonsub/not_match_list.json' ,'w' , encoding='utf-8') as f:
        json.dump(not_match_list , f , indent=4 , ensure_ascii = False)
        print("create jsonfile!!")


print("***********************************************************************************************************************************************")
print("번호 : " , count , "/" , real_count)
print("null값 오류갯수 : " , error_count)
print( "쉬움 : ",easy_count , "중간 : ", medium_count ,"어려움 : ",hard_count , "매우 어려움 : ", extra_hard_count ,  "불일치 : " ,not_match_count)
print("불일치 갯수 ")
print( "쉬움 : ",not_match_easy_count , "중간 : ", not_match_medium_count ,"어려움 : ",not_match_hard_count , "매우 어려움 : ", not_match_extra_hard_count ,  "불일치 총계  : " ,not_match_count)




















