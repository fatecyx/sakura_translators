#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: cyx
# date:  2024/5/9
import argparse
import os
import json
import shutil
import time
from enum import Enum

from collections import OrderedDict, defaultdict
import re
import logging
from pathlib import Path

import requests
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)
CACHE_MAX = 50
ENGINE = 'all'
def init_logger(log_file, stream=True, debug=True):
    # 创建logger
    logger.setLevel(logging.DEBUG)

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if stream:
        # 创建console handler并设置级别为debug
        ch = logging.StreamHandler()
        if debug:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)

        # 添加formatter到ch
        ch.setFormatter(formatter)

        # 添加ch到logger
        logger.addHandler(ch)

    if log_file:
        # 创建FileHandler,指定日志文件路径
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        log_path = Path(log_file)
        file_handler = logging.FileHandler(log_path)
        if debug:
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)

        # 设置日志格式
        file_handler.setFormatter(formatter)

        # 添加FileHandler到logger
        logger.addHandler(file_handler)
    logger.info("测试日志")

re_angle_content = re.compile(r'<(?:[^<>]*?:){0,1}([^<>]*)>')
#re_angle_content = re.compile(r'<[^<>]*?:?([^<>]*)>')
re_angle = re.compile(r'<([^>]*)>')
r1 = re.compile(r"(\<[^\>]*?\>)")
r2_start = re.compile(r"^(\\[a-zA-Z\{\}\\\$\.\|\!\><\^])+")
r2_end = re.compile(r"(\\[a-zA-Z\{\}\\\$\.\|\!\><\^])+$")
re_empty = re.compile(r'(\\[a-zA-Z]|\s)+')

re_engine_split = {
    'rpgmv': [
re.compile(r'en\([^\)]*\)'),
re.compile(r'if\([^\)]*(?:\)|$)'),
re.compile(r'(?:[\\\u001b](?:[!\$\.\{\}\|]|[a-zA-Z]{1,6}\[[^\[\]]*(?:\]|$)|\\?[A-Za-z]))'),
re.compile(r'\%[a-z\d]+'),
re.compile(r'[\u3000＠｜_\s]'),
    ],
    'ks': [
re.compile(r'\[[^\]]*\]')
    ]
}
lst = []
for k, v in re_engine_split.items():
    if k != 'all':
        lst.extend(v)
re_engine_split['all'] = lst

re_split = re.compile(r'(?:\\(?:[!\.\{\}\|]|[a-zA-Z]{1,3}\[[^\[\]]*(?:\]|$)))')
# 正则表达式%1占位符
re_placeholder = re.compile(r'%\d+')

re_script_en = re.compile(r'en\([^\)]*\)')
re_script_if = re.compile(r'if\([^\)]*(?:\)|$)')


japanese_pattern = re.compile(r'[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]') # 匹配日文字符
chinese_pattern = re.compile(r'[\u4E00-\u9FFF]') # 匹配中文字符
korean_pattern = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]') # 匹配韩文字符
english_pattern = re.compile(r'[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]') # 匹配半角和全角英文字母
name_pattern = re.compile(r'([\u30A0-\u30FF―]+)(?:さん|ちゃん|くん|さま|様|先生|女士)') # 匹配片假人名
def add_to_dict(dic, text):
    if not isinstance(text, str) or not text:
        return

    if re_angle_content.search(text):  # 有<>
        lst = re_angle_content.split(text)
        lst = split_by_angle_brackets(text)
        if len(lst)>1:
            for i in lst:
                add_to_dict(dic, i)
            return
    for re_eng in re_engine_split[ENGINE]:
        lst = re_eng.split(text)
        if len(lst) <= 1:
            continue
        for i in lst:
            add_to_dict(dic, i)
        return
    if '\n' in text:
        #lst = text.split('\n')
        lst = re.split(r'\s*\n+\s*', text)
        for i in lst:
            add_to_dict(dic, i)
    else:
        s = remove_newlines_at_ends(text)
        if re_empty.fullmatch(s) or not s:
            return

        elif japanese_pattern.search(s) or chinese_pattern.search(s):
            if s not in dic:
                dic[s] = s

def remove_newlines_at_ends(input_string):
    return input_string.strip().rstrip("【〔「『(（《〈").lstrip("」』】〕）)》〉").strip()
    s = trim_string(input_string)
    # if '（' not in s and s.endswith("）"):
    #     s = s.rstrip("）")
    # return s
    #
    # # 匹配句首和句尾的连续的\\n
    # pattern = r'^[ \x20-\x24\x26-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f\n\u3000\s■◆○▼▲△▽□■▲△▽□■》）】」]+|[\x20-\x28\x2a-\x7f\u3000\n\s■◆○《（【「]+$'
    # # 将匹配的部分替换为空字符串
    # if not isinstance(input_string, str):
    #     print(input_string)
    # result = re.sub(pattern, '', input_string)
    # return result

def split_by_angle_brackets(input_string):
    # 使用正则表达式查找尖括号中的内容

    re_angle = re.compile(r'<([^>]*)>')
    matches = re_angle.finditer(input_string)

    # 遍历匹配项，将字符串分割成子字符串
    last_end = 0
    result = []
    for match in matches:
        start, end = match.span()
        result.extend(input_string[last_end:start].split(','))
        result.append(match.group(1))
        last_end = end
    result.extend(input_string[last_end:].split(','))

    return result


def split_string_with_dict(s, dic):
    if not dic or not s:
        return [s], []
    # 构建正则表达式模式
    pattern = '|'.join(re.escape(key) for key in dic)
    pattern = fr'({pattern})'

    # 使用正则表达式拆分字符串
    parts = []
    start = 0
    words_index_lst = []
    for match in re.finditer(pattern, s):
        parts.append(s[start:match.start()])
        key = match.group(1)
        parts.append(key)
        words_index_lst.append((key, len(parts) - 1))
        # new_dic[key] = len(parts) - 1
        start = match.end()
    if s[start:]:
        parts.append(s[start:])

    return parts, words_index_lst

# if DEBUG:
# dic_output_total = OrderedDict()
# with open("test_translations_result.json", encoding="utf-8") as f:
#     dic_output_total = json.load(f, object_pairs_hook=OrderedDict)
# import translators as ts
# def translate_ja_to_zh(text):
#     results = []
#     for k in text:
#         if k in dic_output_total:
#             results.append(dic_output_total[k])
#         else:
#             print("遗漏", text)
#             results.append(ts.translate_text(text, from_language='ja', to_language='zh'))
#
#     return results

def trim_string(input_string):

    # 存储截取的头尾字符
    head_chars = []
    tail_chars = []
    mid_str = input_string

    # 定义中日文及常用标点符号的正则表达式
    zh_pattern = re.compile(r'[\u4e00-\u9fff]+')
    jp_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3400-\u4dbf\uFF65-\uFF9F]+')
    kuohao = re.compile(r'[「」『』【】〔〕（）()《》〈〉]+')

    # 从头开始检查并截取
    while mid_str and not (zh_pattern.match(mid_str) or jp_pattern.match(mid_str)):
        head_chars.append(mid_str[0])
        mid_str = mid_str[1:]

    # 从尾开始检查并截取
    while mid_str and not (zh_pattern.match(mid_str[-1]) or jp_pattern.match(mid_str[-1])):
        tail_chars.insert(0, mid_str[-1])  # 保持原始顺序插入
        mid_str = mid_str[:-1]
    # return head_chars, mid_str, tail_chars
    # if head_chars or tail_chars:
    #     print(head_chars, mid_str, tail_chars)

    if mid_str:
        head_str = ''.join(head_chars)
        head_mid = re.findall(r'[a-zA-Z0-9０-９Ａ-Ｚａ-ｚ【〔「『(（《〈]+$', head_str)
        if head_mid:
            mid_str = head_mid[0] + mid_str

        punctuation = re.compile(r'^[。．，、；？！」』】〕）)》〉…—…]+')
        tail_str = ''.join(tail_chars)
        tail_mid = punctuation.findall(tail_str)
        if tail_mid:
            mid_str = mid_str + tail_mid[0]

        mid_str = mid_str.lstrip("】〕」』)）》〉・ \u3000").rstrip("【〔「『(（《〈・ \u3000")
    if mid_str!= input_string:
        print(input_string, " => ", mid_str)
    return mid_str

class CacheStatus(Enum):
    NOT_TRANSLATED = 0
    TRANSLATED = 1
    ERROR = 2

DIC_STATUS_INIT = 0
DIC_STATUS_TRANSLATED = 1
DIC_STATUS_ERROR = 2


class Translator:
    def __init__(self, src_file, dict_file, context_lines=10, batch_lines=10, version="0.9"):
        self.base_name = os.path.basename(src_file).rsplit(".",1)[0]
        self.new_file = os.path.join(dst_folder, self.base_name + "_result.txt")
        self.cache_file = os.path.join(dst_folder, self.base_name + "_cache.json")

        self.context_lines = context_lines
        self.batch_lines = batch_lines if batch_lines > 1 else 1

        dic_gpt = None
        if dict_file:
            try:
                dic_gpt = self.load_gpt_dict_list(dict_file)
                logger.info(f"已加载GPT字典{len(dic_gpt)}条")
                print(dic_gpt)
            except Exception as e:
                logger.exception(f"加载GPT字典失败：{e}")

        self.wt = WT(logger, api_url, glossary=dic_gpt, version=version)

        count = 0
        with open(src_file, encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.text_list = [i['src'] for i in self.data_list]

        #self.dic_cache = OrderedDict()
        file_cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, encoding='utf-8') as f:
                lst_cache_data = json.load(f)
            for data in lst_cache_data:
                if data['status'] == DIC_STATUS_TRANSLATED:
                    item = self.data_list[data['index']]
                    if data['src'] != item['src']:
                        logger.warning(f"{self.base_name}缓存数据与源数据(index={data['index']})不匹配：{data['src']}!= {item['src']}")
                        break
                    item['dst'] = data['dst']
                    item['status'] = data['status']
                    count += 1
                elif data['status'] == DIC_STATUS_ERROR:
                    data['status'] = DIC_STATUS_INIT

        print(f"{self.base_name}读取数据{len(self.text_list)}条，缓存{count}条，剩余{len(self.text_list) - count}条")



    def load_gpt_dict_list(self, gpt_dict_path):
        with open(gpt_dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    gpt_dict_data = json_data
                else:
                    gpt_dict_data = list()
                    for src, dst in json_data.items():
                        dict_temp = {
                            "src": src.strip(),
                            "dst": dst.strip()
                        }
                    gpt_dict_data.append(dict_temp)
            except json.JSONDecodeError:
                raw_data = content.splitlines()

                gpt_dict_data = list()
                for raw_data_line in raw_data:
                    raw_data_line = raw_data_line.strip()
                    if not raw_data_line:
                        continue
                    src, temp = raw_data_line.split("->")
                    if "#" in temp:
                        dst, info = temp.split("#")
                        src, dst, info = src.strip(), dst.strip(), info.strip()
                        dict_temp = {
                            "src": src,
                            "dst": dst,
                            "info": info
                        }
                    else:
                        dst = temp.strip()
                        dict_temp = {
                            "src": src.strip(),
                            "dst": dst
                        }
                    gpt_dict_data.append(dict_temp)
        # 按照src的长度排序
        gpt_dict_data = sorted(gpt_dict_data, key=lambda x: len(x['src']), reverse=True)
        return gpt_dict_data

    def save_cache(self):
        if self.cache_count > 0:
            with open(self.cache_file, 'w', encoding='utf-8') as file:
                json.dump(self.data_list, file, indent=4, ensure_ascii=False)

    def _save_result(self, text_index, value, error=False):
        logger.debug(f"翻译成功：{self.data_list[text_index]['src']} => {value}")
        self.data_list[text_index]['dst'] = value
        self.data_list[text_index]['status'] = DIC_STATUS_ERROR if (not value or error) else DIC_STATUS_TRANSLATED

        self.cache_count += 1
        if self.cache_count >= CACHE_MAX:
            self.save_cache()
            self.cache_count = 0

    def join_text(self, lst_words, result_texts, words_index, debug=False):
        new_lst = lst_words.copy()
        for i, item in enumerate(words_index):
            if debug:
                print(i, item)
                print(item[0], "=>", result_texts[i])
            new_lst[item[1]] = result_texts[i]
            # new_lst[sorted_dic[k]] = ""
        return "".join(new_lst)

    def split_text(self, text, debug=False):
        new_dict = {}
        add_to_dict(new_dict, text)
        if debug:
            print(json.dumps(new_dict, indent=4, ensure_ascii=False))
            print("=========")
        # 根据字典分割文本，并生成字典{文本:序号}，序号从0开始，对应原文本的位置，用于后续替换
        lst_words, words_index = split_string_with_dict(text, new_dict)
        if debug:
            print(lst_words)
            print(json.dumps(words_index, indent=4, ensure_ascii=False))
            print("========")
            for k in lst_words:
                print(json.dumps(k, ensure_ascii=False))
            # 根据源文本顺序
            # sorted_dic = sorted(words_index, key=lambda x: x[1])
            print("========")
            print("源文本")
            for k, v in words_index:
                print(v, k)

            print("========")
        # 获取排序后的键列表
        sorted_keys = list(x[0] for x in words_index)
        return lst_words, words_index, sorted_keys

    def main(self):
        self.cache_count = 0
        cache_list = []
        cache_lines = []
        cache_index = 0
        for dic_item in tqdm(self.data_list, total=len(self.data_list), desc=self.base_name, unit='条', mininterval=1):
            text_index = dic_item['index']
            if self.data_list[text_index] != dic_item:
                print(text_index)
                print(self.data_list[text_index])
                print(dic_item)
                raise ValueError("数据不匹配")
            # # 分段调试
            # print(sorted_keys)
            # continue

            if dic_item['status'] != DIC_STATUS_INIT:
                if len(cache_list) > 0:
                    self.translate_cache(cache_index, cache_lines, cache_list)
                    cache_list = []
                    cache_lines = []
                    cache_index = 0
                continue
            if not cache_list:
                cache_index = text_index
                if ENGINE == 'rpgmv':
                    cache_index -= 1
            lst_words, words_index, sorted_keys = self.split_text(dic_item['src'])

            cache_list.append((text_index, lst_words, words_index, len(words_index)))
            cache_lines.extend(sorted_keys)
            if len(cache_list) < self.batch_lines:
                continue

            self.translate_cache(cache_index, cache_lines, cache_list)
            cache_list = []
            cache_lines = []
            cache_index = 0
        if len(cache_list) > 0:
            self.translate_cache(cache_index, cache_lines, cache_list)

    def translate_cache(self, cache_index, cache_lines, cache_list):
        prev_segs = self.text_list[max(cache_index - self.context_lines, 0): max(cache_index, 0)]
        if prev_segs:
            prev_text = "\n".join(prev_segs)

            for re_eng in re_engine_split[ENGINE]:
                prev_text = re_eng.sub('', prev_text)
            prev_segs = prev_text.split("\n")

        try:
            result_texts, error_index_list = self.wt.translate(cache_lines, prev_segs=prev_segs)
        except Exception as e:
            logger.exception(f"翻译失败：{e}")
            raise e
        result_index = 0
        for line_index, lst_words, words_index, length in cache_list:
            # 从result_texts中取出依次对应长度的翻译结果
            lst_result = result_texts[result_index:result_index + length]
            result_index += length
            # 合并翻译结果
            result = self.join_text(lst_words, lst_result, words_index)
            error = False
            for i in error_index_list:
                if result_index <= i < result_index+length:
                    error = True
            self._save_result(line_index, result, error=error)

    def output_result(self, dst_path, delete=False):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, 'w', encoding='utf-8') as file:
            json.dump(self.data_list, file ,indent=4, ensure_ascii=False)
        if delete:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)

class WT:
    def __init__(self, log, endpoint="", glossary=None, test_seg_length=None, version="0.9"):
        self.log = log
        self.seg_length = 500
        self.endpoint = endpoint
        self.version = version
        if test_seg_length is not None:
            self.seg_length = test_seg_length
        self.gpt_dict_data = glossary if glossary else []

    def get_gpt_dict(self, ja: str):
        temp = list()
        for dict_data_simple in self.gpt_dict_data:
            if dict_data_simple['src'] in ja:
                temp.append(dict_data_simple)
        return temp

    def replace_gpt_dict(self, texts: list):
        new_texts = []
        for ja in texts:
            for dict_data_simple in self.gpt_dict_data:
                if dict_data_simple['src'] in ja:
                    ja = ja.replace(dict_data_simple['src'], dict_data_simple['dst'])
            new_texts.append(ja)
        return new_texts

    def translate(self, texts, prev_segs=[]):
        texts = [re.sub(r'[\uff10-\uff19]', lambda x: chr(ord(x.group()) - 65248), text) for text in texts]

        max_length = self.seg_length * 2
        # 从后往前计算prev_segs中的字符串长度，取最多1000个字符长度
        def find_subset_backwards(prev_segs, length):
            total_length = 0
            subset = []
            for s in prev_segs[::-1]:
                if total_length + len(s) <= length:
                    subset.insert(0, s)  # 将字符串插入子集的开头
                    total_length += len(s)
                else:
                    break  # 如果累加长度超过500，停止添加字符串
            return subset

        prev_text = "\n".join(find_subset_backwards(prev_segs, self.seg_length))
        glossary = None
        if self.gpt_dict_data:
            if self.version == "0.10":
                glossary = self.get_gpt_dict("\n".join(texts))
            else:
                texts = self.replace_gpt_dict(texts)
        prompt = "\n".join(texts)
        for i in range(2):
            result = self.translate_prompt(prompt, prev_text, max_length, i > 0, glossary)
            translations = result["text"].replace("<|im_end|>", "").split("\n")
            status = [f"第{i + 1}次"]
            if result["has_degradation"]:
                status.append("退化")
            elif len(texts) != len(translations):
                status.append(f"行数不匹配{len(texts)}!={len(translations)}")
            else:
                status.append("成功")
            self.log.debug(f"翻译结果：{status} {translations}")
            if not result["has_degradation"] and len(texts) == len(translations):
                return translations,[]

        self.log.info("逐行翻译")
        translations = []
        degradation_count = 0
        error_lines = []
        for index, text in enumerate(texts):
            prev_texts = [prev_text] + translations

            result = self.translate_prompt(text, "\n".join(prev_texts), max_length, True, glossary)
            line_translations = result["text"].replace("<|im_end|>", "").split("\n")

            if result["has_degradation"] or len(line_translations) > 1:
                degradation_count += 1
                self.log.warning(f"单行退化{degradation_count}次, 原文：{text} => {line_translations}")
                if degradation_count >= 2:
                    self.log.error("单个分段有2行退化，Sakura翻译器可能存在异常")
                translations.append(line_translations[0])
                error_lines.append(index)
            else:
                translations.append(line_translations[0])

        return translations, error_lines

    def translate_prompt(self, cur_text, prev_text, max_tokens, retry, glossary):
        result = self.create_chat_completions(
            cur_text,
            prev_text,
            {"max_tokens": max_tokens, "frequency_penalty": 0.2 if retry else 0},
            glossary
        )

        text_result = result["choices"][0]["message"]['content']
        has_degradation = result["choices"][0]["finish_reason"] != "stop"
        return {"text": text_result, "has_degradation": has_degradation}

    def create_chat_completions(self, input, prev_text, params, glossary):
        if glossary:
            gpt_dict_text_list = []
            for gpt in glossary:
                src = gpt['src']
                dst = gpt['dst']
                info = gpt['info'] if "info" in gpt.keys() else None
                if info:
                    single = f"{src}->{dst} #{info}"
                else:
                    single = f"{src}->{dst}"
                gpt_dict_text_list.append(single)
            gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
            user_prompt = "根据以下术语表（可以为空）：\n" + gpt_dict_raw_text + "\n\n" + "将下面的日文文本根据上述术语表的对应关系和备注翻译成中文：" + input
        else:
            user_prompt = "将下面的日文文本翻译成中文：" + input
        messages = [
            {"role": "system", "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"},
            {"role": "user", "content": user_prompt}
        ]
        if prev_text:
            messages.insert(1, {"role": "assistant", "content": prev_text})

        data = {
            "model": "",
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.3,
            **params
        }
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(self.endpoint, verify=False, json=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                retry_count += 1
                self.log.error(f"请求失败: {e}. 重试次数: {retry_count}/{max_retries}")
                time.sleep(1)

        raise Exception(f"请求失败,已重试{max_retries}次")


    def replace_glossary(self, text):
        for key, value in self.glossary.items():
            text = text.replace(key, value)
        return text

def func1():
    prev = """夕子はその一番上、
千""".splitlines()
    wt = WT(logger, 'http://86ff-34-41-80-46.ngrok-free.app/v1/chat/completions', None)
    result = wt.translate("""柚は一番下である。""".splitlines(), prev)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print("===============")
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--api_host', type=str, default='http://86ff-34-41-80-46.ngrok-free.app', help='The API host')
    parser.add_argument('--replace_file', type=str, default=None, help='The replace dictionary file')
    parser.add_argument('--src_folder', type=str, default=r"D:\MyDocuments\Python Scripts\py3_test\mtool_trans\text_extract\ks_output\chapter1.txt", help='The source folder')
    parser.add_argument('--dst_folder', type=str, default=r"D:\MyDocuments\Python Scripts\py3_test\mtool_trans\text_extract\ks_result", help='The destination folder')
    parser.add_argument('--engine', type=str, default=r"all", help='all/rpgmv/ks/...for split text,default:all')
    parser.add_argument('--log', type=str, default=r"", help='log file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--silent', action='store_true', help='silent mode')
    parser.add_argument('--delete', action='store_true', help='delete cache after output success')
    parser.add_argument('--context_lines', type=int, default=10, help='处理时参考前文行数,默认为10')
    parser.add_argument('--batch_lines', type=int, default=10, help='一次处理数据行数,默认为10')
    parser.add_argument('--version', type=str, default=r"0.9", help='0.9/0.10, default:0.9')
    parser.add_argument('--zip', type=str, default=None, help='结果目录压缩到')
    args = parser.parse_args()

    init_logger(log_file=args.log, stream=not args.silent, debug=args.debug)

    # Now you can access the values using args.api_host, args.max_len, etc.
    # For example:
    replace_file = args.replace_file
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    api_url = f'{args.api_host}/v1/chat/completions'
    ENGINE = args.engine

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    if os.path.isdir(src_folder):
        file_list = [os.path.join(root, name) for root, dirs, files in os.walk(src_folder) for name in files if name.endswith('.json')]
        # file_list = [x for x in os.listdir(src_folder) if x.endswith('.txt')]
    else:
        file_list = [src_folder]
    if not file_list:
        print(f"没有找到文件{src_folder}")
    for path in file_list:
        dst_path = os.path.join(dst_folder,
                                os.path.relpath(path, src_folder if os.path.isdir(src_folder) else os.path.dirname(src_folder))
                                )
        if os.path.exists(dst_path):
            logger.info(f"文件{dst_path}已存在，跳过")
            continue
        logger.info(f"开始翻译文件{path}")
        trans = Translator(path, replace_file, context_lines=args.context_lines, batch_lines=args.batch_lines, version=args.version)
        try:
            trans.main()
        except:
            logger.exception(f"{path}翻译中出错，保存缓存")
            trans.save_cache()
            break
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            trans.output_result(dst_path, args.delete)
            if args.zip:
                # zip_file(args.zip, dst_folder)
                shutil.make_archive(args.zip, 'zip', dst_folder)
