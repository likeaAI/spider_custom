from datetime import datetime

import json
import os
import sqlglot
import sqlglot.expressions as exp
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView

from django.db import connections
import numpy as np
from elasticsearch import Elasticsearch
from labelon import settings
from job.models import *


### process_sql 필요함수
import sqlite3
from nltk import word_tokenize
import jaydebeapi
import re

# json db용
import numpy as np
import json
from pandas import json_normalize



####### 티베로 json 파일 2개 ##################################################
json_file = "./job/views/api/MANAGE_PHYSICAL_TABLE.json" # table
with open(json_file, 'r', encoding='utf8') as f:
    json_data_table = json.load(f)

json_file = "./job/views/api/MANAGE_PHYSICAL_COLUMN.json" # column
with open(json_file, 'r', encoding='utf8') as e:
    json_data_colum = json.load(e)

df1 = json_normalize(json_data_table)
df2 = json_normalize(json_data_colum)
#############################################################################


class get_column_name(APIView):
    def get(self, request):
        try:
            id=request.GET.get("job_source_id", None)
            if id is None or id == '':
                return Response({"error": "파라미터가 없습니다."})
            with  connections['stats'].cursor() as cursor:
                sql="""select
                            jv.question,
                            jva.answer,
                            j.sql_column_list,
                            j.sql_column_kor_list,
                            j.sql_table_name,
                            j.sql_table_kor_name,
                            da.addition_type
                        from job_visualqa jv
                            inner join job_visualqa_answers jva on jv.id = jva.job_source_id
                            inner join job_file jf on jv.file_id = jf.id
                            inner join job_file_meta j on jf.source_file_id=j.file_id
                inner join dataset_addition da on jv.dataset_id = da.dataset_id                
                        where
                            jv.id={}"""

                cursor.execute(sql.format(id))
                fetch = cursor.fetchall()
                addition_type = ''.join([i[6] for i in fetch])
                query_result = [i[0] for i in fetch]
                answer = [i[1] for i in fetch]
                column_index = self.the_best_column_index([i[2] for i in fetch]) if addition_type == 'BX04' else [i[2] for i in fetch]
                # column_index = [i[2].split('||') for i in fetch] if addition_type == 'BX04' else [i[2] for i in fetch]
                caption_index = self.the_best_column_index([i[3] for i in fetch]) if addition_type == 'BX04' else [i[3] for i in fetch]
                table_name = self.the_best_column_index([i[4] for i in fetch]) if addition_type == 'BX04' else [i[4] for i in fetch]
                table_kor_name = self.the_best_column_index([i[5] for i in fetch]) if addition_type == 'BX04' else [i[5] for i in fetch]

                column_index_list = column_index if addition_type == 'BX04' else column_index[0].split(',')
                # column_index_list = [column_index[0][0].split(','), column_index[0][1].split(',')] if addition_type == 'BX04' else column_index[0].split(',')
                caption_index_list = caption_index if addition_type == 'BX04' else caption_index[0].split(',')
                if addition_type == 'BX04':
                    col_list = [{"column": col, "name": caption_index_list[idx]} for idx, col in np.ndenumerate(column_index_list)]
                else:
                    col_list = [{"column": col, "name": caption_index_list[i]} for i, col in enumerate(column_index_list)]

            columns = []
            column_list = []
            condition_list = []
            for select in sqlglot.parse_one(query_result[0]).find_all(exp.Select):
                if (addition_type == 'BX04' and select.parent is None) or addition_type !='BX04':
                    for col_exp in select.select().expressions:
                        select_col_list = []
                        if col_exp.key == 'subquery':
                            select_col_list = col_exp.this.expressions
                        elif col_exp.key != 'column':
                            # select절에 집계함수가 있을경우 key가 column인 데이터를 append한다.
                            select_col_list.append(col_exp.this)
                        else:
                            select_col_list.append(col_exp)
                        for col in select_col_list:
                            if not col.this.sql() in columns:
                                columns.append(col.this.sql())
                                try:
                                    index = np.where(column_index_list == col.this.sql().upper()) if addition_type == 'BX04' else column_index_list.index(col.this.sql())
                                    if addition_type == 'BX04':
                                        column_index = str(index[0][0])+'-'+str(index[1][0])
                                    else:
                                        column_index = index
                                    startindex, endindex, tagText = self.compare_tokenize(answer[0], caption_index_list[index], addition_type)
                                    column_name_kor = caption_index_list[index][0] if addition_type == 'BX04' else caption_index_list[index]
                                except ValueError as e:
                                    index = None
                                column_list.append({"columnName": col.this.sql(), "columnNameKor": column_name_kor, "columnIndex": column_index, "startIndex": startindex, "endIndex": endindex, "sqlTokenType": "BY01"})

            col=""
            def get_cond(condition):
                conditions = []
                if condition is None:
                    return
                for con in condition.find_all(exp.Condition):
                    # not in, is null, is not null 조건절 : 태깅할 조건값이 없으니 패스
                    if con.key in ['not', 'is'] or (con.parent.key == 'not' and con.key == 'is'):
                        continue
                    elif con.key == 'in':
                        # not in, in 조건절 : 태깅할 조건값들을 넘겨준다.
                        for sub_con in con.expressions:
                            conditions.append(sub_con)
                    elif (con.parent.key == 'in' and con.key == 'column') or (con.parent.key == 'select' and con.key == 'column'):
                        # todo: subquery인지 알아야한다. 그다음에 해당 where절만 찾아서 conditions에 넣는다.
                        test_con = con
                    elif con.key in ['and', 'or', 'column', 'literal', 'null']:
                        # subquery다음에 빈 expresstion이 온다.
                        if con.expression is None:
                            continue
                        # elif con.expression.key == 'subquery':
                        #     tttt_ = con.expression
                        else:
                            get_cond(con.expression)
                    else:
                        if con.expression is None or con.expression.key == 'subquery':
                            continue
                        else:
                            conditions.append(con)
                return conditions

            for where in sqlglot.parse_one(query_result[0]).find_all(exp.Where):
                condition = get_cond(where)
                for cond in condition:
                    # in조건절: cond.key=literal / 나머지 con.key= Column
                    col = cond.find(exp.Literal).parent.this.sql() if cond.key == 'literal' else cond.find(exp.Column).this.sql()
                    # col_test = cond.parent_select.selects
                    if cond.key in ['is']:
                        val = cond.expression.sql()
                    else:
                        if cond.key == 'like':
                            val = cond.find(exp.Literal).this.replace('%','')
                        else:
                            # literal이 없는 경우 아마.. 조건값이 subquery?!
                            if cond.find(exp.Literal) is None:
                                # 조건값이 없으면 태깅하지않는다.
                                continue
                            else:
                                val = cond.find(exp.Literal).this
                    try:
                        index = answer[0].index(val)
                    except ValueError as e:
                        index = None
                    except Exception as e:
                        return Response({"error": str(e)})

                    try:
                        con_index = np.where(column_index_list == col) if addition_type == 'BX04' else column_index_list.index(col)
                        if addition_type == 'BX04':
                            con_index = str(con_index[0][0]) + "-" + str(con_index[1][0])
                    except ValueError as e:
                        con_index = None
                    except Exception as e:
                        return Response({"error": str(e)})

                    startindex,endindex, tagText = self.compare_tokenize(answer[0], val, addition_type)
                    # 조건값이 지문에 없으면 태깅을 안 한다.
                    if startindex is None or endindex is None:
                        continue

                    condition_list.append({"columnIndex": con_index, "tagText": tagText, "tagStartIndex": startindex, "tagEndIndex": endindex, "tagType": "BP02", "sqlTokenType": "BY02"})

            return Response({"question": answer[0], "query": query_result[0], "select": column_list, "conditions": condition_list,"tableName" : table_name, "tableNameKor" : table_kor_name, "columns": col_list})
        except Exception as e:
            return  Response({"error": str(e)})




    def compare_tokenize(self, question, value, addition_type):
        try:
            is_date_type = False
            question = question.upper()
            if addition_type == 'BX04' and type(value) != str:
                value = value[0]
            try:
                test_t = value[:8] if addition_type == 'BX04' or len(value) == 14 else value
                value = str(datetime.strptime(test_t, '%Y%m%d').date())
                is_date_type = True
            except ValueError as e:
                print("Incorrect data format({0}), should be date type".format(test_t))

            source = list(question)
            constr = []
            start_index, end_index = None, None
            if is_date_type:
                date_list = value.split('-')
                start_index = question.find(date_list[0])
                end_index = question.find(str(int(date_list[-1]))+'일')
                q_index = end_index + question[start_index:end_index].count(' ')
                end_index = q_index+1 if question[q_index] == '일' else q_index
                value = question[start_index: end_index]
                return start_index, end_index, value
            else:
                target = list(value.replace(' ','').upper())

                for index, word in enumerate(source):
                    if str(constr).upper() == value.replace(' ','').upper():
                        break
                    for t_word in target:
                        if word == ' ':
                            if not start_index is None:
                                end_index = end_index + 1
                            break
                        if word == t_word:
                            constr.append(target.pop(0))
                            if start_index is None:
                                start_index = index
                                # 한글자라도 end_index는 start_index보다 크다.
                                end_index = index + 1
                            else:
                                end_index = end_index + 1
                            break
                        else:
                            if not str(constr).upper() == value.upper():
                                start_index = None
                                end_index = None
                                target=list(value.replace(' ','').upper())

            return start_index, end_index, value
        except Exception as e:
            print(e)
            return None,None

    def the_best_column_index(self, column_index):
        for item in column_index:
            col_list1 = list(item.split('||')[0].split(','))
            col_list2 = list(item.split('||')[1].split(','))
            abs_len = abs(len(col_list2) - len(col_list1))
            for i in range(abs_len):
                if len(col_list1) > len(col_list2):
                    col_list2.append(None)
                else:
                    col_list1.append(None)

            # if abs_len > 0:
            #     rows = []
            #     for a in [col_list1, col_list2]:
            #         rows.append(np.pad(a, (0, abs_len), 'constant', constant_values=0)[:abs_len])
            #     test_ = np.concatenate(rows, axis=0).reshape(-1, abs_len)
            #     return np.array([col_list1, col_list2], (0, abs_len), 'constant', constant_values=None)
            # else:
            #     return np.array([col_list1, col_list2])
            return np.array([col_list1, col_list2])


class get_image_list(APIView):
    def get(self, request):
        es = Elasticsearch(settings.ELASTICSEARCH_HOST)
        try:
            page = int(request.GET.get("current_page", 1))
            page_size = int(request.GET.get("page_size", 78))
            keyword = request.GET.get("keyword", None)

            e = es.search(index='job_image_search', body={"from": (page * page_size)-(page_size - 1) , "size": page_size, "query": {
                "multi_match": {"query": keyword, "fields": ["category", "search_keyword"]}}})
            result = {
                "numFound": e['hits']['total']['value'],
                "docs": [i["_source"] for i in e['hits']['hits']],
                "start": page
            }
            return Response(
                {"response": 200, "result": result}, 200)
        except Exception as e:
            return Response({"response": 400, "error": str(e)}, 400)


class set_image_path(APIView):
    def get(self, request):
        es = Elasticsearch(settings.ELASTICSEARCH_HOST)
        try:
            index = 'job_image_search'
            job_id = request.GET.get('job_id', None)
            category = request.GET.get('category', None)
            search_keyword = request.GET.get('keyword', None)
            job_image = JobImage.objects.get(id=job_id)

            if category == '':
                try:
                    category = job_image.file.tags
                except:
                    Response({"result": 401, "error" : "카테고리 / 키워드가 없습니다."}, 401)
            if search_keyword == '':
                try:
                    search_keyword = job_image.file.jobfilemeta2jobfile.image_description
                except:
                    Response({"result": 401, "error" : "카테고리 / 키워드가 없습니다."}, 401)


            doc = {
                "id": job_image.file.id,
                "file_path": job_image.file.file_path,
                "file_name" : job_image.file.file_name,
                "category" : category,
                "search_keyword" : search_keyword,
                "annotator_id": job_image.annotator_id
            }

            result = es.search(index=index,body={"query": {"match": {"id": str(job_image.file.id)}}})

            if len(result['hits']['hits']) == 0:
                es.index(index= index, doc_type="_doc", body=doc)
            return Response(
                {"result": result}, 200)
        except Exception as e:
            return Response({"result" : 400,"error": str(e)}, 400)

class spider_restapi(APIView) : # 티베로 클래스 내부에서 접속, jar 파일 경로 설정 필요
    def get(self, request) :
        try:
            conn = jaydebeapi.connect(
                "com.tmax.tibero.jdbc.TbDriver",
                "jdbc:tibero:thin:@172.7.0.23:8629:tibero",
                ["labelon", "euclid!@)$labelon"],
                "./job/views/api/tibero6-jdbc.jar", # <- jar 파일 경로를 설정해고 해결함
            )
            cur = conn.cursor()

            query_string = request.GET.get('query_string', None)
            #print(query_string , type(query_string))
            level_data = spider_level_query(cur, query_string) # 쿼리입력시 난이도 및 유형 판별 함수 2개의 str 유형, 난이도를 반환
            query_score = level_data[1]
            query_level = level_data[0]

            cur.close()

            return Response({"score" : query_score , "level" : query_level})

        except Exception as e :
            print(e)
            return Response({"recheck table_name or column name" :  query_string})


class spider_restapi2(APIView) : # tibero data 마리아db로 복사하여 장고db등록 후 사용
    def get(self, request ) :
        with connections['tibero'].cursor() as cur : # 장고 백엔드 db에서 커서가져오는 명령어
            try:

                query_string = request.GET.get('query_string', None)
                #print(query_string , type(query_string))
                level_data = spider_level_query(cur, query_string)
                query_score = level_data[1]
                query_level = level_data[0]

                return Response({"score" : query_score , "level" : query_level})

            except Exception as e :
                print(e)
                return Response({"recheck table_name or column name" :  query_string})


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists' )
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}

def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks): # 여기서 join table도 같이 조회가 되어야 함
    tables = scan_alias(toks) # as로 명명한 테이블들 없으면 공란 []

    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key

    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)


    #print('toks', toks[idx]) # 해당정보 매우 중요
    #print('tables_with_alias', tables_with_alias)
    #print('start_idx' , start_idx)


    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx



################################################## evalution.py 추출 함수 ##############################################
# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True

def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested

def get_keywords(sql):
    res = set() # 중복점수를 허용하지 않는다. where ,groupby, having이든 하나만 인정
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])



def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN 1개는 0점 ... 2개는 1점.
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count



def eval_hardness(sql): # 기존 깃허브 오리지널 판정코드
     count_comp1_ = count_component1(sql)
     count_comp2_ = count_component2(sql)
     count_others_ = count_others(sql)

     if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
         return "easy"
     elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
             (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
         return "medium"

     elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
             (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) \
             or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
         return "hard"
     else:
         return "extra hard"



# def eval_hardness(sql): # gitlab 수정 코드 확인
#     count_comp1_ = count_component1(sql)
#     count_comp2_ = count_component2(sql)
#     count_others_ = count_others(sql)
#
#     if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
#         return "easy"
#     elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
#             (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
#         return "medium"
#     elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
#             (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
#             (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1) :
#         return "hard"
#     else:
#         return "extra hard"


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print('column_names_original: ', column_names_original)
        #print('table_names_original: ', table_names_original)
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i
        return idMap

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {} # 전체 형태
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original'] # 컬럼 갯수
        table_names_original = db['table_names_original'] # 테이블명
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original} # db_id에 해당하는 컬럼과 테이블명
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


########################################################################################################################
def sql_level(query, db_id) : # db정보를 json으로 가져와서 비교하는 함수
    sql = query
    db_id = db_id
    table_file = "C:/Users/yoonsub/test05.json" # cp949 에러,메모장에서 anis다시 인코딩시 한글은 사라짐

    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]

    print("테이블 : " , table)
    print("스키마차이 : ", schema)

    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)

    comp_1 = count_component1(sql_label)
    comp_2 = count_component2(sql_label)
    comp_other = count_others(sql_label)

    level = eval_hardness(sql_label)
    score = f"{comp_1}-{comp_2}-{comp_other}"

    print("query : ", sql)
    print(f"type : {score} , level : {level}")

    return str(level), str(score)

def spider_level_query_json(query):  # 티베로 manage.physical_table , column 을 json으로 읽고 작동
    query = query.replace('\n', ' ')
    query = query.upper()
    query = re.sub(' +', ' ', query)  # 다중공백을 변환
    query_list = query.split(" ")
    table_index = query_list.index("FROM") + 1
    table_name = str(query_list[table_index])

    print(table_index, table_name)
    if "JOIN" in query_list:  # 테이블2개 join이 있다면
        #global table2_name
        # print("join exist")
        table2_index = query_list.index("JOIN") + 1
        table2_name = str(query_list[table2_index])

        data_pd = df1['LOGICAL_TABLE_ENGLISH'] == table_name  # 쿼리에서 from 절 이후 테이블명을 받아 데이터프레임에서 조회
        db_id = df1['ID'][data_pd]
        db_index = db_id.index[0]  # 결과값이 true인 인덱스를 반환

        column_pd = df2['DATA_PHYSICAL_ID'] == db_id[db_index]
        column_list = df2['LOGICAL_COLUMN_ENGLISH'][column_pd]

        join_data_pd = df1['LOGICAL_TABLE_ENGLISH'] == table2_name
        db_id_02 = df1['ID'][join_data_pd]
        db_index_02 = db_id_02.index[0]

        join_column_pd = df2['DATA_PHYSICAL_ID'] == db_id_02[db_index_02]
        column_list2 = df2['LOGICAL_COLUMN_ENGLISH'][join_column_pd]

        # SPIDER Schema 클래스에서 사용할수있는 형태로 가공
        cols = []
        for i in column_list:
            i = i.lower()
            cols.append(i)

        cols2 = []
        for i in column_list:
            i = i.lower()
            cols2.append(i)

        column_names_original = []
        for col in column_list:
            col = col.upper()
            for j in [0]:
                add_col = [j, col]
                column_names_original.append(add_col)

        for col in column_list2:
            col = col.upper()
            for j in [1]:
                add_col = [j, col]
                column_names_original.append(add_col)

        first_col = [[-1, '*']]
        column_names_original = first_col + column_names_original
        # table_names_original = [table_name]
        table_names_original2 = [table_name, table2_name]

        # schemas 형태
        schemas = {db_id[db_index]: {table_name.lower(): cols,
                                     table2_name.lower(): cols2}}  # <- table with alie 대문자,소문자 차이는 여기서 보정
        tables = {db_id[db_index]: {"column_names_original": column_names_original,
                                    "table_names_original": table_names_original2}}

        # 점수 및 난이도 판별 구간
        sql = query
        schema = schemas[db_id[db_index]]
        table = tables[db_id[db_index]]

        schema = Schema(schema, table)  # class Schema 를 사용 오리지널 컬럼과 , 오리지널 테이블을 입력 데이터가 중요
        sql_label = get_sql(schema, sql)

        comp_1 = count_component1(sql_label)
        comp_2 = count_component2(sql_label)
        comp_other = count_others(sql_label)

        level = eval_hardness(sql_label)
        score = f"{comp_1}-{comp_2}-{comp_other}"

        return str(level), str(score)

    else:  # TABEL이 하나라면
        data_pd = df1['LOGICAL_TABLE_ENGLISH'] == table_name

    db_id = df1['ID'][data_pd]
    db_index = db_id.index[0]

    column_pd = df2['DATA_PHYSICAL_ID'] == db_id[db_index]
    column_list = df2['LOGICAL_COLUMN_ENGLISH'][column_pd]

    cols = []
    for i in column_list:
        i = i.lower()
        cols.append(i)

    column_names_original = []
    for col in column_list:
        col = col.upper()
        for j in [0]:
            add_col = [j, col]
            column_names_original.append(add_col)

    first_col = [[-1, '*']]
    column_names_original = first_col + column_names_original
    table_names_original = [table_name]

    # schemas 형태
    schemas = {db_id[db_index]: {table_name.lower(): cols}}  # <- table with alie 대문자,소문자 차이는 여기서 보정
    tables = {
        db_id[db_index]: {"column_names_original": column_names_original, "table_names_original": table_names_original}}
    print(tables)

    sql = query
    schema = schemas[db_id[db_index]]
    table = tables[db_id[db_index]]

    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)

    comp_1 = count_component1(sql_label)
    comp_2 = count_component2(sql_label)
    comp_other = count_others(sql_label)

    level = eval_hardness(sql_label)
    score = f"{comp_1}-{comp_2}-{comp_other}"

    return str(level), str(score)


def spider_level_query(cur, query) : # db에 접속해서 쿼리문만으로 테이블명과 db_id를 가져와서 쿼리 난이도를 판별하는 함수

    # 쿼리를 공백 분리 후 FROM 인덱스로 테이블명 확인 / join table 값 확인 필요
    query = query.replace('\n', ' ')
    query = query.upper()
    query = re.sub(' +', ' ', query)  # 다중공백을 변환
    query_list = query.split(" ")
    table_index = query_list.index("FROM") + 1
    table_name = str(query_list[table_index])

    if "JOIN" in query_list :
        global table2_name
        #print("join exist")
        table2_index = query_list.index("JOIN") + 1
        table2_name = str(query_list[table2_index])

        # 확인한 테이블명으로 data_sql 쿼리문 실행하여 db_id, physical_data_id등을 조회
        data_sql = f"SELECT PHYSICAL_TABLE_NAME,id,DATA_BASIC_ID FROM MANAGE_PHYSICAL_TABLE WHERE LOGICAL_TABLE_ENGLISH = '{table_name}'"
        # print(data_sql)
        cur.execute(data_sql)
        data01 = cur.fetchall()

        db_id = data01[0][1]
        # physical_data_id = data01[0][2] # 컬럼조회용으로 physical_data_id 사용
        # print(physical_data_id)
        #print(db_id)

        column_sql = f"SELECT LOGICAL_COLUMN_ENGLISH FROM MANAGE_PHYSICAL_COLUMN WHERE DATA_PHYSICAL_ID = '{db_id}'"
        cur.execute(column_sql)
        column_list = cur.fetchall()  # pyysical_data_id로  해당 컬럼 모두 조회

        # join 용 테이블 명 및 컬럼정보 조회
        join_data_sql = f"SELECT PHYSICAL_TABLE_NAME,id,DATA_BASIC_ID FROM MANAGE_PHYSICAL_TABLE WHERE LOGICAL_TABLE_ENGLISH = '{table2_name}'"
        cur.execute(join_data_sql)
        data02 = cur.fetchall()

        db_id_02 = data02[0][1]
        # physical_data_id = data01[0][2] # 컬럼조회용으로 physical_data_id 사용
        # print(physical_data_id)
        #print(db_id_02)

        join_column_sql = f"SELECT LOGICAL_COLUMN_ENGLISH FROM MANAGE_PHYSICAL_COLUMN WHERE DATA_PHYSICAL_ID = '{db_id_02}'"
        cur.execute(join_column_sql)
        column_list2 = cur.fetchall()  # pyysical_data_id로  해당 컬럼 모두 조회

        # 분석용 sql구문 전처리
        cols = []
        for i in column_list:
            i = str(i).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            i = i.lower()
            cols.append(i)
        #print(cols)

        cols2 = []
        for i in column_list2:
            i = str(i).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            i = i.lower()
            cols2.append(i)

        #print("join 테이블 컬럼 : ", cols2)
        # column_names_original create
        column_names_original = []
        for col in column_list:
            col = str(col).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            col = col.upper()
            for j in [0]:
                add_col = [j, col]
                column_names_original.append(add_col)

        for col in column_list2:
            col = str(col).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            col = col.upper()
            for j in [1]:
                add_col = [j, col]
                column_names_original.append(add_col)

        #print('column_names_original', column_names_original)

        first_col = [[-1, '*']]
        column_names_original = first_col + column_names_original
        table_names_original = [table_name]
        table_names_original2 = [table_name, table2_name]

        #print('table_names_original2', table_names_original2)

        # schemas 형태
        schemas = {
            db_id: {table_name.lower(): cols, table2_name.lower(): cols2}}  # <- table with alie 대문자,소문자 차이는 여기서 보정
        tables = {
            db_id: {'column_names_original': column_names_original, "table_names_original": table_names_original2}}

        sql = query
        schema = schemas[db_id]
        table = tables[db_id]

        schema = Schema(schema, table)  # class Schema 를 사용 오리지널 컬럼과 , 오리지널 테이블을 조건을 바꿔야한다.
        sql_label = get_sql(schema, sql)

        comp_1 = count_component1(sql_label)
        comp_2 = count_component2(sql_label)
        comp_other = count_others(sql_label)

        level = eval_hardness(sql_label)
        score = f"{comp_1}-{comp_2}-{comp_other}"

        #print("query : ", sql)
        #print(f"type : {score} , level : {level}")

        return str(level), str(score)


    else :
        #print("join not exist")
        # 확인한 테이블명으로 data_sql 쿼리문 실행하여 db_id, physical_data_id등을 조회
        data_sql = f"SELECT PHYSICAL_TABLE_NAME,id,DATA_BASIC_ID FROM MANAGE_PHYSICAL_TABLE WHERE LOGICAL_TABLE_ENGLISH = '{table_name}'"
        # print(data_sql)
        cur.execute(data_sql)
        data01 = cur.fetchall()

        db_id = data01[0][1]
        # physical_data_id = data01[0][2] # 컬럼조회용으로 physical_data_id 사용
        # print(physical_data_id)
        # print(db_id)

        column_sql = f"SELECT LOGICAL_COLUMN_ENGLISH FROM MANAGE_PHYSICAL_COLUMN WHERE DATA_PHYSICAL_ID = '{db_id}'"
        cur.execute(column_sql)
        column_list = cur.fetchall()  # pyysical_data_id로  해당 컬럼 모두 조회

        # 분석용 sql구문 전처리
        cols = []
        for i in column_list:
            i = str(i).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            i = i.lower()
            cols.append(i)
        # print(cols)

        # column_names_original create
        column_names_original = []
        for col in column_list:
            col = str(col).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            col = col.upper()
            for j in [0]:
                add_col = [j, col]
                column_names_original.append(add_col)
        # print(column_names_original)

        first_col = [[-1, '*']]
        column_names_original = first_col + column_names_original
        table_names_original = [table_name]

        # schemas 형태
        schemas = {db_id: {table_name.lower(): cols}}  # <- table with alie 대문자,소문자 차이는 여기서 보정
        tables = {db_id: {"column_names_original": column_names_original, "table_names_original": table_names_original}}
        # print(tables)

        sql = query
        schema = schemas[db_id]
        table = tables[db_id]

        schema = Schema(schema, table)  # class Schema 를 사용 오리지널 컬럼과 , 오리지널 테이블을 조건을 바꿔야한다.
        sql_label = get_sql(schema, sql)

        comp_1 = count_component1(sql_label)
        comp_2 = count_component2(sql_label)
        comp_other = count_others(sql_label)

        level = eval_hardness(sql_label)
        score = f"{comp_1}-{comp_2}-{comp_other}"

        #print("query : ", sql)
        #print(f"type : {score} , level : {level}")

        return str(level), str(score)