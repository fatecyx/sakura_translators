#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: cyx
# date:  2024/5/9
import argparse
import os
import json
import shutil
import time

import pandas as pd
from collections import OrderedDict, defaultdict
import re
import logging
from pathlib import Path

import requests
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logger = logging.getLogger(__name__)
CACHE_MAX = 100

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


re_angle = re.compile(r'<([^>]*)>')
r1 = re.compile(r"(\<[^\>]*?\>)")
r2_start = re.compile(r"^(\\[a-zA-Z\{\}\\\$\.\|\!\><\^])+")
r2_end = re.compile(r"(\\[a-zA-Z\{\}\\\$\.\|\!\><\^])+$")
re_empty = re.compile(r'(\\[a-zA-Z]|\s)+')


re_split = re.compile(r'(?:\\(?:[!\.\{\}\|]|[a-zA-Z]{1,3}\[[^\[\]]*(?:\]|$)))')
re_split_signs = re.compile(r'\(|\)|\-|\/|\:')
# 正则表达式%1占位符
re_placeholder = re.compile(r'%\d+')

re_script_en = re.compile(r'en\([^\)]*\)')
re_script_if = re.compile(r'if\([^\)]*(?:\)|$)')

re_talk_lst = [
    re.compile(r'^([^\n「」\r\n]+)\n「([^」]*)(?:」\s*)?$'),
    re.compile(r'^([^\n（）\r\n]+)\n（([^）]*)(?:）\s*)?$'),
]

set_names = defaultdict(int)

japanese_pattern = re.compile(r'[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]') # 匹配日文字符
chinese_pattern = re.compile(r'[\u4E00-\u9FFF]') # 匹配中文字符
korean_pattern = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]') # 匹配韩文字符
english_pattern = re.compile(r'[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]') # 匹配半角和全角英文字母
name_pattern = re.compile(r'([\u30A0-\u30FF―]+)(?:さん|ちゃん|くん|さま|様|先生|女士)') # 匹配片假人名
def add_to_dict(dic, text):
    if not isinstance(text, str) or not text:
        return

    if re_angle.findall(text):  # 有<>
        lst = split_by_angle_brackets(text)
        for i in lst:
            if not i or r2_start.fullmatch(i):
                continue
            elif i.startswith("/"):
                continue
            if ":" in i:
                for j in i.split(":"):
                    add_to_dict(dic, j)
            else:
                add_to_dict(dic, i)
    elif re_split.search(text):
        lst = re_split.split(text)
        for i in lst:
            add_to_dict(dic, i)
    elif re_script_if.search(text):
        lst = re_script_if.split(text)
        for i in lst:
            add_to_dict(dic, i)
    elif re_script_en.search(text):
        lst = re_script_en.split(text)
        for i in lst:
            add_to_dict(dic, i)
    elif re_placeholder.search(text):
        for i in re_placeholder.split(text):
            print(text,' => ', i)
            add_to_dict(dic, i)
    # elif re_split_signs.search(text):
    #     for i in re_split_signs.split(text):
    #         print("SIGNS_SPLIT: ", text,' => ', i)
    #         add_to_dict(dic, i)
    elif '\n' in text:
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
    return input_string.strip()
    s = trim_string(input_string)
    if '（' not in s and s.endswith("）"):
        s = s.rstrip("）")
    return s
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
#
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
class Translator:
    def __init__(self, src_file):
        self.base_name = os.path.basename(src_file).replace(".json", "")
        self.new_file = os.path.join(dst_folder, self.base_name + "_trans.json")
        self.cache_file = os.path.join(dst_folder, self.base_name + "_cache.json")
        self.err_file = os.path.join(dst_folder, self.base_name + "_error.json")

        self.wt = WT(logger, api_url, replace_file)

        with open(src_file, encoding='utf-8') as f:
            src_data = json.load(f, object_pairs_hook=OrderedDict)
        self.data = OrderedDict()
        self.dic_new = OrderedDict()
        self.dic_cache = OrderedDict()

        self.dic_err = OrderedDict()

        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, encoding='utf-8') as f:
                self.dic_cache = json.load(f, object_pairs_hook=OrderedDict)

        # 从self.data中去掉self.dic_cache
        count = 0
        for k, v in src_data.items():
            if k not in self.dic_cache:
                self.data[k] = v
            else:
                self._save_result(k, self.dic_cache[k], is_cache=True)
                count += 1
        if count > 0:
            print(f"读取缓存{count}条，剩余{len(self.data)}条")

    def save_cache(self):
        if self.cache_count > 0:
            with open(self.cache_file, 'w', encoding='utf-8') as file:
                json.dump(self.dic_cache, file, indent=4, ensure_ascii=False)

    def _save_result(self, original_text, value, is_cache=False):

        if not value:
            self.dic_err[original_text] = value
            return

        if original_text != value:
            if value.endswith("。") and not original_text.endswith(tuple("。！？.,!?")):
                value = value.rstrip("。")

        self.dic_new[original_text] = value

        # 替换字典中替换失败的情况
        for k, v in self.wt.glossary.items():
            if k in original_text and v not in value:
                self.dic_err[original_text] = value
                break

        if is_cache:
            return

        self.dic_cache[original_text] = value

        self.cache_count += 1
        if self.cache_count >= CACHE_MAX:
            self.cache_count = 0
            with open(self.cache_file, 'w', encoding='utf-8') as file:
                json.dump(self.dic_cache, file, indent=4, ensure_ascii=False)

    def split_and_translate(self, text, debug=False):
        # 提取翻译文本new_dict
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
        # trans_text = "\n".join(sorted_keys)
        try:
            result_texts = self.wt.translate(sorted_keys)
            #result_texts = translate_ja_to_zh(sorted_keys)
            try:
                logger.debug(f"{sorted_keys} => {result_texts}")
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"翻译失败：{e}")
            return None

        new_lst = lst_words.copy()
        for i, item in enumerate(words_index):
            if debug:
                print(i, item)
                print(item[0], "=>", result_texts[i])
            new_lst[item[1]] = result_texts[i]
            # new_lst[sorted_dic[k]] = ""

        return "".join(new_lst)

    def main(self):
        self.cache_count = 0
        for original_text in tqdm(self.data, total=len(self.data), desc=self.base_name, unit='条', mininterval=1):
            if original_text in self.dic_cache:
                self.dic_new[original_text] = self.dic_cache[original_text]
                continue

            result = self.split_and_translate(original_text)
            self._save_result(original_text, result)

    def output_result(self, is_delete=False):
        with open(self.new_file, 'w', encoding='utf-8') as file:
            json.dump(self.dic_new, file, indent=4, ensure_ascii=False)
        if self.dic_err:
            with open(self.err_file, 'w', encoding='utf-8') as file:
                json.dump(self.dic_err, file, indent=4, ensure_ascii=False)
                print("有错误", len(self.dic_err))
        if is_delete and os.path.exists(self.cache_file):
            os.remove(self.cache_file)


class WT:
    def __init__(self, log, endpoint="", glossary=None, test_seg_length=None):
        self.log = log
        self.seg_length = 500
        self.endpoint = endpoint

        if test_seg_length is not None:
            self.seg_length = test_seg_length
        if glossary:
            if not os.path.exists(glossary):
                self.log.error(f"glossary文件{glossary}不存在")
            else:
                with open(glossary, encoding='utf-8') as f_replace:
                    dic_replace = json.load(f_replace)
                    # 根据value长度排序，保证替换顺序
                    self.glossary = OrderedDict(sorted(dic_replace.items(), key=lambda x: len(x[1]), reverse=True))
        else:
            self.glossary = {}

    def translate(self, texts, prev_segs=[]):
        texts = [re.sub(r'[\uff10-\uff19]', lambda x: chr(ord(x.group()) - 65248), text) for text in texts]
        texts = [self.replace_glossary(text) for text in texts]

        max_length = self.seg_length * 2
        prompt = "\n".join(texts)
        prev_text = "\n".join(prev_segs[-int(1000 / self.seg_length):])

        for i in range(2):
            result = self.translate_prompt(prompt, prev_text, max_length, i > 0)
            translations = result["text"].replace("<|im_end|>", "").split("\n")
            status = [f"第{i + 1}次"]
            if result["has_degradation"]:
                status.append("退化")
            elif len(texts) != len(translations):
                status.append("行数不匹配")
            else:
                status.append("成功")

            if not result["has_degradation"] and len(texts) == len(translations):
                return translations

        self.log.info("逐行翻译")
        translations = []
        degradation_count = 0
        for text in texts:
            prev_texts = [prev_text] + translations
            result = self.translate_prompt(text, "\n".join(prev_texts), max_length, True)
            if result["has_degradation"]:
                degradation_count += 1
                self.log.warning(f"单行退化{degradation_count}次, 原文：{text} => {result['text']}")
                if degradation_count >= 2:
                    raise Exception("单个分段有2行退化，Sakura翻译器可能存在异常")
                translations.append(text)
            else:
                translations.append(result["text"].replace("<|im_end|>", ""))

        return translations

    def translate_prompt(self, prompt, prev_text, max_tokens, retry):
        result = self.create_chat_completions(
            prompt,
            prev_text,
            {"max_tokens": max_tokens, "frequency_penalty": 0.2 if retry else 0}
        )

        text = result["choices"][0]["message"]['content']
        has_degradation = result["choices"][0]["finish_reason"] != "stop"
        return {"text": text, "has_degradation": has_degradation}

    def create_chat_completions(self, prompt, prev_text, params):
        messages = [
            {"role": "system", "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"},
            {"role": "user", "content": "将下面的日文文本翻译成中文：" + prompt}
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--api_host', type=str, default='http://127.0.0.1:5000', help='The API host')
    parser.add_argument('--replace_file', type=str, default=None, help='The replace dictionary file')
    parser.add_argument('--src_folder', type=str, default=r"D:\ACGN\AiNiee\src", help='The source folder')
    parser.add_argument('--dst_folder', type=str, default=r"D:\ACGN\AiNiee\dst", help='The destination folder')
    parser.add_argument('--log', type=str, default=r"/kaggle/working/log.txt", help='log file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--silent', action='store_true', help='silent mode')
    parser.add_argument('--zip', type=str, default=None, help='结果目录压缩到')
    parser.add_argument('--delete', action='store_true', help='delete cache after output success')

    args = parser.parse_args()

    init_logger(log_file=args.log, stream=not args.silent, debug=args.debug)

    # Now you can access the values using args.api_host, args.max_len, etc.
    # For example:
    api_host = args.api_host
    replace_file = args.replace_file
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    api_url = f'{api_host}/v1/chat/completions'


    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    if os.path.isdir(src_folder):
        file_list = [x for x in os.listdir(src_folder) if x.endswith('.json')]
    else:
        file_list = [src_folder]

    for path in file_list:
        logger.info(f"开始翻译文件{path}")
        trans = Translator(path)
        try:
            trans.main()
        except:
            logger.exception(f"{path}翻译中出错，保存缓存")
            trans.save_cache()
            break
        else:
            trans.output_result(args.delete)
            if args.zip:
                # zip_file(args.zip, dst_folder)
                shutil.make_archive(args.zip, 'zip', dst_folder)
    # with open(r'test_translations.json', 'w', encoding='utf-8') as fp:
    #     json.dump(dic_output_total, fp, indent=4, ensure_ascii=False)