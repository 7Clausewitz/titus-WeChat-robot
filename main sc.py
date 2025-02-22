from wxauto import *
import os
import json
import sqlite3
from datetime import datetime
from openai import OpenAI
import threading
import queue
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import json
import random
from datetime import datetime, timedelta
import sys
import requests

# 初始化 API 客户端
def call_deepseek_api(prompt, model, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API 调用失败: {response.status_code}, {response.text}")

# 获取当前时间戳并格式化
timestamp = time.time()
local_time = time.localtime(timestamp)
formatted_time = time.strftime("%Y-%m-%d %H:%M", local_time)


if os.path.exists('config.json'):
    # 读取配置文件
    with open('config.json', 'r',encoding='utf-8') as configfile:
        config = json.load(configfile)
else:
    config = {
        'DATABASE':{
            'api_key': 'API_COPY_HERE',
            'base_url': 'https://api.siliconflow.cn/v1/chat/completions',
            'master': 'root',
            'model_set': '你的模型名字',
            'prompt_set':'AI模型设定',
            'called_name':'泰图斯',
            'auto_calling_you_window':'文件传输助手',
            'auto_call':1
        }
    }
    with open('config.json', 'w',encoding='utf-8') as configfile:
        json.dump(config, configfile, indent=4)
    with open('config.json', 'r',encoding='utf-8') as configfile:
        config = json.load(configfile)
    




# 获取配置信息
API = config['DATABASE']['api_key']
URL = config['DATABASE']['base_url']
MASTER = config['DATABASE']['master']
MOD_S = config['DATABASE']['model_set']
Pro_set = config['DATABASE']['prompt_set']
called_name =config['DATABASE']['called_name']
calling_you_window =config['DATABASE']['calling_you_window']
auto_call =config['DATABASE']['auto_call']


if os.path.exists('listen_list.json'):
    with open('listen_list.json','r',encoding='utf-8') as configlist:
        con_l_list = json.load(configlist)
else:
        # 监听列表和群组列表
    listen_list = [
        '文件传输助手'
    ]
    with open('listen_list.json','w',encoding='utf-8') as configlist:
        json.dump(listen_list,configlist)
    with open('listen_list.json','r',encoding='utf-8') as configlist:
        con_l_list = json.load(configlist)

if os.path.exists('group_list.json'):
    with open('group_list.json','r',encoding='utf-8') as configlist:
        con_g_list = json.load(configlist)
else:
    group_list = [

    ]
    with open('group_list.json','w',encoding='utf-8') as configlist:
        json.dump(group_list,configlist)
    with open('group_list.json','r',encoding='utf-8') as configlist:
        con_g_list = json.load(configlist)

listen_list = con_l_list
group_list = con_g_list









master = MASTER
model_set = MOD_S # 模型设置
prompt_set = Pro_set+'/n附加提示：你的主人是' + master

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def print_colored(text, color):
    """
    在控制台中打印带有颜色的文本。
    
    参数:
    text (str): 要打印的文本。
    color (str): 颜色代码，例如 'red', 'green', 'blue', 'yellow' 等。
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'end': '\033[0m'
    }
    print(f"{colors[color]}{text}{colors['end']}")



def remove_first_two_empty_lines_alt(text):
    """移除文本前两行空行"""
    lines = text.splitlines()
    i = 0
    empty_count = 0
    while i < len(lines) and empty_count < 2:
        if lines[i].strip() == '':
            empty_count += 0
        else:
            break
        i += 1
    return '\n'.join(lines[i:])


def extract_text_after_keyword(text, keyword):
    """从文本中提取关键词后的内容"""
    index = text.find(keyword)
    if index != -1:
        # 找到关键词，提取其后的文本
        return text[index + len(keyword):].strip()
    return None


def remove_keywords(text, keywords):
    """
    此函数用于在给定的文本中搜索关键字并将其删除。

    :param text: 要处理的原始字符串
    :param keywords: 包含要删除的关键字的列表
    :return: 移除关键字后的字符串
    """
    for keyword in keywords:
        text = text.replace(keyword, "")
    return text


# 数据库初始化 --------------------------------------------------
def init_database():
    """初始化数据库表结构"""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()

    # 用户消息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 用户画像表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            username TEXT PRIMARY KEY,
            analysis_history TEXT DEFAULT '[]',
            current_profile TEXT DEFAULT '{}',
            last_analyzed_id INTEGER DEFAULT 0,
            analysis_count INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()


# 消息存储系统 --------------------------------------------------
class MessageStore:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._init_worker()

    def _init_worker(self):
        """后台存储工作线程"""
        def worker():
            while True:
                task = self.task_queue.get()
                try:
                    self._store_message(task['user'], task['message'])
                except Exception as e:
                    print(f"存储失败: {str(e)}")
                finally:
                    self.task_queue.task_done()

        threading.Thread(target=worker, daemon=True).start()

    def async_store(self, username, message):
        """异步存储消息"""
        self.task_queue.put({
            'user': username,
            'message': message
        })

    def _store_message(self, username, message):
        """消息存储与触发分析"""
        with sqlite3.connect('memory.db') as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('BEGIN IMMEDIATE')

                # 初始化用户记录
                cursor.execute('''
                    INSERT OR IGNORE INTO user_profiles (username) 
                    VALUES (?)
                ''', (username,))

                # 插入新消息
                cursor.execute('''
                    INSERT INTO user_messages (username, message)
                    VALUES (?, ?)
                ''', (username, message))

                # 获取当前最新消息ID
                cursor.execute('''
                    SELECT MAX(id) FROM user_messages WHERE username = ?
                ''', (username,))
                latest_id = cursor.fetchone()[0]

                # 计算未分析消息数
                cursor.execute('''
                    SELECT last_analyzed_id FROM user_profiles 
                    WHERE username = ?
                ''', (username,))
                last_analyzed = cursor.fetchone()[0]

                unanalyzed_count = latest_id - last_analyzed

                # 触发条件判断
                if unanalyzed_count >= 12:
                    # 计算实际需要分析的批次
                    batches = unanalyzed_count // 12
                    for i in range(batches):
                        batch_max_id = last_analyzed + (i + 1) * 12
                        AnalysisEngine().request_analysis(username, batch_max_id)

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e


# 分析引擎 ------------------------------------------------------
class AnalysisEngine:
    _instance = None
    lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls.lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.task_queue = queue.PriorityQueue()
                    cls._instance._init_worker()
        return cls._instance

    def _init_worker(self):
        """分析工作线程"""
        def worker():
            while True:
                priority, username = self.task_queue.get()
                try:
                    self._analyze_user(username)
                except Exception as e:
                    print(f"分析失败: {str(e)}")
                finally:
                    self.task_queue.task_done()

        for _ in range(3):  # 3个工作线程
            threading.Thread(target=worker, daemon=True).start()

    def request_analysis(self, username, priority=1):
        """提交分析请求"""
        self.task_queue.put((priority, username))

    def _analyze_user(self, username):
        """执行分析的核心逻辑"""
        with sqlite3.connect('memory.db') as conn:
            cursor = conn.cursor()

            # 获取分析基准点
            cursor.execute('''
                SELECT current_profile, last_analyzed_id 
                FROM user_profiles 
                WHERE username = ?
            ''', (username,))
            profile_data = cursor.fetchone()

            current_profile = json.loads(profile_data[0]) if profile_data and profile_data[0] else {}
            last_id = profile_data[1] if profile_data else 0

            # 获取新消息
            cursor.execute('''
                SELECT id, message FROM user_messages
                WHERE username = ?
                ORDER BY id DESC
                LIMIT 12
            ''', (username,))
            new_messages = cursor.fetchall()[::-1]  # 倒序恢复时间顺序
            # 消息验证
            if len(new_messages) != 12:
                print(f"消息不足12条，当前：{len(new_messages)}")
                return

            # 类型校验
            try:
                # 提取消息ID和内容
                msg_ids = [int(msg[0]) for msg in new_messages]
                message_texts = [str(msg[1]) for msg in new_messages]
                min_id, max_id = min(msg_ids), max(msg_ids)
            except (ValueError, IndexError) as e:
                print(f"消息格式错误：{str(e)}")
                return

            # 检查是否已处理
            cursor.execute('SELECT last_analyzed_id FROM user_profiles WHERE username = ?', (username,))
            last_analyzed = cursor.fetchone()[0] or 0
            if last_analyzed >= min_id:
                print(f"消息{min_id}-{max_id}已处理过")
                return
            not_group = True
            if username and username[0].lower() == 's':
                origin_username = username[4:]
                for group_name in group_list:
                    if group_name in origin_username:
                        analysis_prompt = self._build_group_prompt(current_profile, new_messages)
                        not_group = False
                    else:
                        not_group = True
                if not_group:
                    # 生成分析提示
                    analysis_prompt = self._build_prompt(current_profile, new_messages)
            elif username and username[0].lower() == 'f':
                origin_username = username[6:]
                for group_name in group_list:
                    if group_name in origin_username:
                        analysis_prompt = self._build_group_prompt(current_profile, new_messages)
                        not_group = False
                    else:
                        not_group = True
                if not_group:
                    # 生成分析提示
                    analysis_prompt = self._build_prompt(current_profile, new_messages)
            else:
                origin_username = username[:-2]
                analysis_prompt = self._build_ai_memory_prompt(current_profile, new_messages)


            # 调用AI接口
            try:
                new_profile = self._get_ai_analysis(analysis_prompt)
                self._update_profile(cursor, username, new_profile, max_id)
                conn.commit()
                restart_program()
            except Exception as e:
                conn.rollback()
                print(f"分析失败：{str(e)}")
                raise

    def _build_prompt(self, current_profile, new_messages):
        """构建分析提示"""
        # 安全提取消息内容
        message_texts = []
        for msg in new_messages:
            if len(msg) >= 2 and isinstance(msg[1], str):
                message_texts.append(msg[1])
            else:
                message_texts.append("[内容格式异常]")

        return f"""
        历史用户画像（JSON格式）：
        {json.dumps(current_profile, indent=2, ensure_ascii=False)}

        新增的12条消息：
        {"-" * 20}
        {chr(10).join(message_texts)}
        {"-" * 20}
        请根据以下规则生成新的用户画像：
        1. 用户的名称
        2. 保留长期稳定的心理学特征（如持续3次以上出现的特征）
        3. 更新短期变化特征（如最近出现的兴趣点，或者最近提到的事情）
        4. 运用心理学知识进行分析
        5. 冲突时以最新数据为准

        请严格按照以下要求生成JSON：
        1. 必须包含所有指定字段
        2. 不使用任何注释或额外文本
        3. 确保双引号闭合
        4. 确保没有多余文本,也不要有任何多余的换行
        生成格式：
        {{
            "name": "名称（字符串）",
            "core_traits": ["特征1", "特征2"],
            "interests": {{"类别": ["具体项"]}},
            "communication_style": "描述（字符串）",
            "last_updated": "日期（字符串）"
        }}
        """

    def _build_group_prompt(self, current_profile, new_messages):
        """构建分析提示"""
        # 安全提取消息内容
        message_texts = []
        for msg in new_messages:
            if len(msg) >= 2 and isinstance(msg[1], str):
                message_texts.append(msg[1])
            else:
                message_texts.append("[内容格式异常]")

        return f"""
        这是一个群聊的档案：
        历史群聊中用户画像（JSON格式）：
        {json.dumps(current_profile, indent=2, ensure_ascii=False)}

        新增的12条消息：
        {"-" * 20}
        {chr(10).join(message_texts)}
        {"-" * 20}
        请根据以下规则生成新的用户画像：
        1. 不同用户的名称
        2. 不同用户的行为特征（如持续3次以上出现的特征）
        3. 不同用户的兴趣点（如最近出现的兴趣点，或者最近提到的事情）
        4. 运用心理学知识进行分析
        5. 冲突时以最新数据为准

        请严格按照以下要求生成JSON：
        1. 必须包含所有指定字段
        2. 不使用任何注释或额外文本
        3. 确保双引号闭合
        4. 确保没有多余文本,也不要有任何多余的换行
        生成格式：
        {{
            "name": "名称1/名称2（字符串）",
            "core_traits": ["名称1/特征1", "名称2/特征2"],
            "interests": {{"名称1": ["具体项"]}},
            "communication_style": "名称1：风格/名称2：风格（字符串）",
            "last_updated": "日期（字符串）"
        }}
        """

    def _build_ai_memory_prompt(self, current_profile, new_messages):
        """构建分析提示"""
        # 安全提取消息内容
        message_texts = []
        for msg in new_messages:
            if len(msg) >= 2 and isinstance(msg[1], str):
                message_texts.append(msg[1])
            else:
                message_texts.append("[内容格式异常]")

        return f"""
        这是一个聊天中ai的消息记录
        历史聊天中的消息记录总结（JSON格式）：
        {json.dumps(current_profile, indent=2, ensure_ascii=False)}

        新增的12条消息：
        {"-" * 10}
        {chr(10).join(message_texts)}
        {"-" * 10}
        请根据以下规则生成新的聊天记录画像，以精简的总结发生了什么事情：
        1. 这个聊天场景名称的概括
        2. 聊天中的关键词
        3. 聊天的具体内容
        4. 自我的评价
        5. 冲突时以最新数据为准

        请严格按照以下要求生成JSON：
        1. 必须包含所有指定字段
        2. 不使用任何注释或额外文本
        3. 确保双引号闭合
        4. 确保没有多余文本,也不要有任何多余的换行
        生成格式：
        {{
            "name": "名称（字符串）",
            "core_traits": ["关键词1", "关键词2"],
            "interests": {{"名称1": ["具体项"]}},
            "communication_style": "自我评价（字符串）",
            "last_updated": "日期（字符串）"
        }}
        """
        

    def _get_ai_analysis(self, prompt):
        """调用 AI 接口"""
        try:
            response = call_deepseek_api(prompt, model_set, API)
            print("进行了分析后," + called_name + "觉得：" + response['choices'][0]['message']['content'])
            return json.loads(response['choices'][0]['message']['content'])
        except Exception as e:
            print(f"AI 分析失败: {str(e)}")
            return {}

    def _update_profile(self, cursor, username, new_profile, last_msg_id):
        profile_json = json.dumps(new_profile, ensure_ascii=False)
        """更新用户画像并记录最后分析的消息ID"""
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (username, analysis_history, current_profile, last_analyzed_id, analysis_count)
            VALUES (
                ?,
                COALESCE(
                    json_insert(
                        (SELECT analysis_history FROM user_profiles WHERE username = ?), 
                        '$[#]', 
                        json(?)
                    ), 
                    json_array(?)
                ),
                ?,
                ?,
                COALESCE((SELECT analysis_count FROM user_profiles WHERE username = ?), 0) + 1
            )
        ''', (
            username,  # 主键username
            username,  # SELECT子查询中的username
            profile_json,  # 新画像JSON对象
            profile_json,  # 初始数组值
            json.dumps(profile_json, ),  # 当前画像（完整JSON）
            last_msg_id,  # 最后分析的消息ID（关键修改点）
            username  # 计数子查询的username
        ))


# 记忆调用系统 --------------------------------------------------
class MemorySystem:
    @lru_cache(maxsize=100)
    def get_profile(self, username):
        """获取当前用户画像（带缓存）"""
        with sqlite3.connect('memory.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT current_profile FROM user_profiles
                WHERE username = ?
            ''', (username,))
            result = cursor.fetchone()
            return json.loads(result[0]) if result and result[0] else {}

    def get_recent_messages(self, username, limit=8):
        """获取近期消息"""
        with sqlite3.connect('memory.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT message FROM user_messages
                WHERE username = ?
                ORDER BY id DESC
                LIMIT ?
            ''', (username, limit))
            return [msg[0] for msg in cursor.fetchall()[::-1]]


# 主服务接口 ----------------------------------------------------
class ChatService:
    def __init__(self):
        self.store = MessageStore()
        self.memory = MemorySystem()
        init_database()

    def process_message(self, username, message,origin_username):
        """处理用户消息的全流程"""
        # 异步存储消息
        self.store.async_store(username, message)

        # 获取记忆上下文
        context = self._build_context(username,origin_username)
        not_group = True
        for group_name in group_list:
            if group_name in username:
                answer = self._group_reply(context, message)
                not_group = False
            else:
                not_group = True
        if not_group:
            answer = self._generate_reply(context, message)

        answer = remove_first_two_empty_lines_alt(answer)
        self.store.async_store(origin_username+'ai', answer)
        answer = answer.split('%')

        for send_one in answer:
            print(called_name+'说：' + send_one)
            chat.SendMsg(send_one)

        # 生成回复
        return

    def posscess_auto_message(self,who):
        context = service._build_context('self'+who,who)
        answer = self._auto_ask_(context)
        answer = remove_first_two_empty_lines_alt(answer)
        self.store.async_store(who+'ai', answer)
        answer = answer.split('%')
        for send_one in answer:
            print(called_name+'说：' + send_one)
            wx.SendMsg(send_one,who)


    def _build_context(self, username,origin_username):
        """构建记忆上下文"""
        if username and username[0].lower() == 's':
            username_ob = 'friend' + origin_username
            long_term_ob = self.memory.get_profile(username_ob)
            short_term_ob = self.memory.get_recent_messages(username_ob, limit=4)
        else:
            username_ob = 'self' + origin_username
            long_term_ob = self.memory.get_profile(username_ob)
            short_term_ob = self.memory.get_recent_messages(username_ob, limit=4)
        
        return {
            'long_term': self.memory.get_profile(username),
            'short_term': self.memory.get_recent_messages(username),
            'user': username,
            'long_term_ob': long_term_ob,
            'short_term_ob': short_term_ob,
            'short_term_ai':self.memory.get_recent_messages(origin_username+'ai',limit=12),
            'long_term_ai':self.memory.get_profile(origin_username+'ai')
        }

    def _generate_reply(self, context, message):
        """生成带上下文的回复"""
        prompt = f"""
请参考用户档案与画像，同时参考设定进行回答
用户在发生信息时，会以”name：message time“的形式发送（阅读信息时请参考）
在回复时，请你像真人使用网络聊天工具一样，一条一条发送信息，请用%符号分段你要说的话,文字中间不要有空的换行。（示例：你好，你今天怎么样%有什么我可以帮助你的吗？）
语言风格：
1. 你使用口语进行表达，比如会使用一些语气词和口语连接词，如“嗯、啊、当然、那个”，等来增强口语风格。
2. 你可以将动作、神情语气、心理活动、故事背景放在（）中来表示，为对话提供补充信息。
一、主用户档案：[
{json.dumps(context['long_term'], indent=2, ensure_ascii=False)}]

二、主用户最近对话：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term'])])}]

三、另一个在聊天中的用户的聊天记录（你是两人聊天中的第三者）,请同时参考：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ob'])])}]

四、另一个在聊天中的用户的画像档案（你是两人聊天中的第三者）,请同时参考：[
{json.dumps(context['long_term_ob'], indent=2, ensure_ascii=False)}]

五、在聊天中，你回复的消息记录,请同时参考：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ai'])])}]

六、在聊天中，你回复的消息记录的总结,请同时参考：[
{json.dumps(context['long_term_ai'], indent=2, ensure_ascii=False)}]


七、当前消息：[
{message}]

        """
        print("分析后发送给" + called_name + "：" + prompt)
        try:
            response = call_deepseek_api(prompt, model_set, API)
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"群聊回复生成失败: {str(e)}")
            return "抱歉，我暂时无法生成回复。"

    def _group_reply(self, context, message):
        """生成带上下文的回复"""

        prompt = f"""
请注意这是一个组群聊消息的信息，其中一个档案是{master}的档案，一个是这个群聊中其他多个用户的档案（你会在开头看见他们的名字），当不同的人呼叫你时（参考当前信息），你将针对这个用户发出的信息进行回答
请参考用户档案与画像，同时参考设定进行回答
用户在发生信息时，会以”name：message time“的形式发送（阅读信息时请参考）
在回复时，请你像真人使用网络聊天工具一样，一条一条发送信息，请用%符号分段你要说的话,文字中间不要有空的换行。（示例：你好，你今天怎么样%有什么我可以帮助你的吗？）
语言风格：
1. 你使用口语进行表达，比如会使用一些语气词和口语连接词，如“嗯、啊、当然、那个”，等来增强口语风格。
2. 你可以将动作、神情语气、心理活动、故事背景放在（）中来表示，为对话提供补充信息。
一、主用户档案：[
{json.dumps(context['long_term'], indent=2, ensure_ascii=False)}]

二、主用户最近对话：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term'])])}]

三、另一个在聊天中的用户的聊天记录（你是群组聊天中的独立个体，不是其中任何一个人）,请同时参考进行互动：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ob'])])}]

四、另一个在聊天中的用户的画像档案（你是两人聊天中的第三者）,请同时参考：[
{json.dumps(context['long_term_ob'], indent=2, ensure_ascii=False)}]

五、在聊天中，你回复的消息记录,请同时参考：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ai'])])}]

六、在聊天中，你回复的消息记录的总结,请同时参考：[
{json.dumps(context['long_term_ai'], indent=2, ensure_ascii=False)}]

七、当前消息：[
{message}]

        """
        print("分析后发送给" + called_name + "：" + prompt)
        try:
            response = call_deepseek_api(prompt, model_set, API)
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"群聊回复生成失败: {str(e)}")
            return "抱歉，我暂时无法生成回复。"

    def _auto_ask_(self, context):
        prompt = f"""
请参考用户档案与画像，同时参考设定，发出主动寻找已知用户聊天的消息(可以是打招呼，问候，关系，提起新话题，但请不要重复用户和自己的话语)
用户在发生信息时，会以”name：message time“的形式发送（阅读信息时请参考）
在回复时，请你像真人使用网络聊天工具一样，一条一条发送信息，请用%符号分段你要说的话,文字中间不要有空的换行。（示例：你好，你今天怎么样%有什么我可以帮助你的吗？）
语言风格：
1. 你使用口语进行表达，比如会使用一些语气词和口语连接词，如“嗯、啊、当然、那个”，等来增强口语风格。
2. 你可以将动作、神情语气、心理活动、故事背景放在（）中来表示，为对话提供补充信息。
一、主用户档案：[
{json.dumps(context['long_term'], indent=2, ensure_ascii=False)}]

二、主用户最近对话：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term'])])}]

三、另一个在聊天中的用户的聊天记录（你是群组聊天中的独立个体，不是其中任何一个人）,请同时参考进行互动：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ob'])])}]

四、另一个在聊天中的用户的画像档案（你是两人聊天中的第三者）,请同时参考：[
{json.dumps(context['long_term_ob'], indent=2, ensure_ascii=False)}]

五、在聊天中，你回复的消息记录,请同时参考：[
{chr(10).join([f"{i + 1}. {msg}" for i, msg in enumerate(context['short_term_ai'])])}]

六、在聊天中，你回复的消息记录的总结,请同时参考：[
{json.dumps(context['long_term_ai'], indent=2, ensure_ascii=False)}]


        """
        print("分析后发送给" + called_name + "：" + prompt)
        try:
            response = call_deepseek_api(prompt, model_set, API)
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"群聊回复生成失败: {str(e)}")
            return "抱歉，我暂时无法生成回复。"



listen_mark = False

def set_listen_mark_periodically():
    global listen_mark
    while True:
        current_time = time.localtime()
        current_hour = current_time.tm_hour

        # 检查是否在早上 8 点到晚上 12 点之间
        if 8 <= current_hour < 24:
            # 检查是否是每两小时的起始点
            if current_hour % 4 == 0:
                # 计算距离当前小时结束还剩多少秒
                seconds_to_end_of_hour = 3600 - (current_time.tm_min * 60 + current_time.tm_sec)
                # 生成一个随机秒数
                random_second = random.randint(0, seconds_to_end_of_hour)
                # 等待到随机的秒数
                time.sleep(random_second)
                # 设置 listen_mark 为 True
                listen_mark = True
                # 可以在这里添加一些逻辑来处理 listen_mark 为 True 的情况
                # 例如，触发某个事件或执行某个函数
                print(f"At {time.strftime('%Y-%m-%d %H:%M:%S', current_time)}, listen_mark is set to True.")
                # 等待到下一个两小时的开始
                remaining_seconds = seconds_to_end_of_hour - random_second
                time.sleep(remaining_seconds)
            else:
                # 若不是两小时的起始点，计算到下一个两小时起始点还需等待的秒数
                next_two_hour_start = (current_hour // 4 + 1) * 4
                seconds_to_wait = (next_two_hour_start - current_hour) * 3600 - (current_time.tm_min * 60 + current_time.tm_sec)
                time.sleep(seconds_to_wait)
        else:
            # 若不在 8 点到 24 点之间，计算到早上 8 点还需等待的时间
            next_8am = time.mktime((current_time.tm_year, current_time.tm_mon, current_time.tm_mday + 1 if current_hour >= 8 else current_time.tm_mday, 8, 0, 0, 0, 0, 0))
            time_to_wait = next_8am - time.time()
            time.sleep(time_to_wait)



print_colored("=" * 42, 'green')
print_colored(called_name+" - 智能微信聊天助手", 'green')
print_colored("=" * 42, 'green')
# 获取当前微信客户端

try:
    wx = WeChat()
except Exception as e:
    error_msg = f"笨蛋，未登录微信: {e}"
    print(error_msg)


# 初始化监听列表
for i in listen_list:
    wx.AddListenChat(who=i, savepic=True)




commands = [
    '听令，增加人物',
    '听令，增加群组',
    '听令，删除群组',
    '听令，删除人物',
    '听令，列出列表',
    '听令，修改问候窗口',
    '听令，修改设定'
]


if __name__ == "__main__":
    # 初始化服务
    service = ChatService()
    wait = 1  # 设置1秒查看一次是否有新消息
    thread = threading.Thread(target=set_listen_mark_periodically)
    thread.daemon = True  # 设置为守护线程，主程序退出时线程也会退出
    thread.start()
    while True:


        if auto_call == 1:
            if listen_mark:
                listen_mark = False
                print(str(calling_you_window))
                tauto = threading.Thread(target=service.posscess_auto_message, args=(calling_you_window,))
                tauto.start()
                tauto.join()

        msgs = wx.GetListenMessage()
        for chat in msgs:
            who = chat.who  # 获取聊天窗口名（人或群名）
            one_msgs = msgs.get(chat)  # 获取消息内容
            for msg in one_msgs:
                msgtype = msg.type  # 获取消息类型
                content = msg.content  # 获取消息内容，字符串类型的消息内容
                sender = msg.sender
            if content.startswith('听令，'):
                    # 接入设置的判断（可扩展为菜单）
                find_addset = extract_text_after_keyword(content, '听令，增加人物')
                find_delset = extract_text_after_keyword(content, '听令，删除人物')
                find_addgset = extract_text_after_keyword(content, '听令，增加群组')
                find_delgset = extract_text_after_keyword(content, '听令，删除群组')
                find_showlist = extract_text_after_keyword(content, '听令，列出列')
                find_showcm = extract_text_after_keyword(content, '听令，列出命')
                find_heywindow = extract_text_after_keyword(content, '听令，修改问候窗口')
                find_prompt = extract_text_after_keyword(content, '听令，修改设定')

                if find_addset:
                    try:
                        listen_list.append(find_addset)
                        print("现在人物列表：" + str(listen_list))
                        with open('listen_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(listen_list, configlist)
                        chat.SendMsg(str(listen_list))
                        for i in listen_list:
                            wx.AddListenChat(who=i, savepic=True)
                    except Exception as e:
                        error_msg = f"执行 '听令，增加人物' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)

                if find_delset:
                    try:
                        listen_list.remove(find_delset)
                        with open('listen_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(listen_list, configlist)
                        print("现在人物列表：" + str(listen_list))
                        chat.SendMsg(str(listen_list))
                        for i in listen_list:
                            wx.AddListenChat(who=i, savepic=True)
                    except Exception as e:
                        error_msg = f"执行 '听令，删除人物' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)

                if find_addgset:
                    try:
                        group_list.append(find_addgset)
                        print("现在群组列表：" + str(group_list))
                        listen_list.append(find_addgset)
                        with open('group_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(group_list, configlist)
                        with open('listen_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(listen_list, configlist)
                        chat.SendMsg(str(group_list))
                    except Exception as e:
                        error_msg = f"执行 '听令，增加群组' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)

                if find_delgset:
                    try:
                        group_list.remove(find_delgset)
                        print("现在群组列表：" + str(group_list))
                        listen_list.remove(find_delgset)
                        with open('group_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(group_list, configlist)
                        with open('listen_list.json', 'w', encoding='utf-8') as configlist:
                            json.dump(listen_list, configlist)
                        chat.SendMsg(str(group_list))
                    except Exception as e:
                        error_msg = f"执行 '听令，删除群组' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)

                if find_showlist:
                    try:
                        print("现在人物列表：" + str(listen_list) + "\n现在群组列表：" + str(group_list))
                        chat.SendMsg("现在人物列表：" + str(listen_list) + "\n现在群组列表：" + str(group_list))
                    except Exception as e:
                        error_msg = f"执行 '听令，列出列表' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)

                if find_showcm:
                    try:
                        for i, command in enumerate(commands, start=1):
                            print(f"{i}. {command}")
                            chat.SendMsg(f"{i}. {command}")
                    except Exception as e:
                        error_msg = f"执行 '听令，列出命令' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)
                
                if find_heywindow:
                    try:
                        with open('config.json','r',encoding='utf-8') as configlist:
                            config=json.load(configlist)
                        config['DATABASE']['calling_you_window'] = find_heywindow
                        calling_you_window = find_heywindow
                        with open('config.json','w',encoding='utf-8') as configlist:
                            json.dump(config,configlist,indent=4)
                        chat.SendMsg('heywindow已切换为：'+str(calling_you_window))
                        listen_mark = True
                    except Exception as e:
                        error_msg = f"执行 '听令，修改问候窗口' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)
                if find_prompt:
                    try:
                        with open('config.json','r',encoding='utf-8') as configlist:
                            config=json.load(configlist)
                        config['DATABASE']['prompt_set'] = find_prompt
                        Pro_set = find_prompt
                        with open('config.json','w',encoding='utf-8') as configlist:
                            json.dump(config,configlist,indent=4)
                        chat.SendMsg('设定已切换为：'+str(find_prompt))
                    except Exception as e:
                        error_msg = f"执行 '听令，切换设定' 操作时出错: {e}"
                        print(error_msg)
                        chat.SendMsg(error_msg)


            if content.startswith(str(called_name)):  # 唤醒判断
                find_ask = extract_text_after_keyword(content, str(called_name)+'，')
                if find_ask:
                    print("finding_asking")
                    if msgtype == 'self':
                        selfmsg = 'self' + who
                        find_ask = master + ':' + find_ask + "\n    说话时间：" + formatted_time
                        ts = threading.Thread(target=service.process_message, args=(selfmsg,find_ask,who))
                        ts.start()
                        ts.join()

                    if msgtype == 'friend':
                        friendmsg = 'friend' + who
                        if who in group_list:
                            find_ask = sender + ':' + find_ask + "\n     说话时间：" + formatted_time
                            tf = threading.Thread(target=service.process_message, args=(friendmsg, find_ask,sender))
                            tf.start()
                        else:
                            find_ask = who + ':' + find_ask + "\n     说话时间：" + formatted_time
                            tf = threading.Thread(target=service.process_message, args=(friendmsg, find_ask,sender))
                            tf.start()
                            ts.join()

        time.sleep(wait)