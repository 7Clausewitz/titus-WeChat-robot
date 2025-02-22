# titus-WeChat-robot
# 智能会话助手 - 微信集成解决方案

## 项目概述
本解决方案通过wxauto库实现微信本地机器人，提供拟真对话体验与智能记忆管理。系统采用模块化设计，具备上下文感知、多级记忆存储和动态画像分析等核心功能，有效平衡隐私保护与智能交互需求。

## 一、核心功能体系

### 1. 交互引擎
- **场景无感调用**：通过特定唤醒词(如"泰图斯")在任意聊天界面触发助手
- **拟真消息模拟**：
  - 自然分段发送
  - 支持表情符号与格式修饰
  - 主动随机聊天（默认关闭，开启见教程）
  - 在代码中已经预设了优质关键词，不会调prompt的小白也能轻松使用

### 2. 记忆管理系统
#### 短期记忆库：
##### ai会记录在该窗口你对ai的聊天记录和对方（群组）对ai的聊天信息。为了隐私安全，ai只会读取你呼唤他的消息记录（如：泰图斯，xxx），不会记录其他日常信息。

```python
def _store_message(self, username, message):
    """消息存储与隐私过滤机制"""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        # 用户白名单初始化
        cursor.execute('''
            INSERT OR IGNORE INTO user_profiles (username) 
            VALUES (?)
        ''', (username,))
        # 消息指纹去重
        msg_hash = hashlib.md5(message.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO user_messages (username, message, hash)
            SELECT ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM user_messages 
                WHERE hash = ?
            )
        ''', (username, message, msg_hash, msg_hash))
```
#### 长期画像分析：
##### ai会在你（或者在聊天框的对方）对他的聊天信息达到8条时，分析你的人物画像。记忆你的性格，特质，兴趣爱好。同时ai会在自己回答12条消息时，总结聊天内容。减少了上下文Token使用。
```python
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
            new_messages = cursor.fetchall()[::-1]  # 倒序恢复时间顺序
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
```
 
### 3.自适应交互协议
#### 上下文感知模型
    详见def _generate_reply(self, context, message)函数，可自行diy
    
### 4.微信快捷命令：可以直接在微信上发消息进行修改设置（详见教程），可以自行diy更多命令。命令编写逻辑为：
```python
            if content.startswith('听令，'): 为真
                find_XXXX = extract_text_after_keyword(content, '听令，你的命令名字')为真
            if find_XXXX:为真
                # 执行里面的命令逻辑
```
## 二、使用教程
文件基于硅基流动的api调用，如需要使用其他模型，请修改源码中调用api的def call_deepseek_api 函数

### 1.硅基流动注册：https://cloud.siliconflow.cn/i/HAOHHJPf
注册后会赠送大概100万token

### 2.硅基流动模型广场：https://cloud.siliconflow.cn/models


### 3.创建api：
![image](https://github.com/user-attachments/assets/df8a466d-b3eb-4f48-bc71-6417bd0cbfbe)


### 4.用记事本打开config.json文件
![image](https://github.com/user-attachments/assets/2189d75e-886d-4c5f-9075-85bc8eb11435)


### 5.复制你要选择的模型（见官网文档，小白选deepseek-ai/DeepSeek-V3就对了）：

      如：
      deepseek-ai/DeepSeek-R1,
      deepseek-ai/DeepSeek-V3,
      deepseek-ai/DeepSeek-R1-Distill-Llama-70B, 
      deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, 
      eepseek-ai/DeepSeek-R1-Distill-Qwen-14B, 
      deepseek-ai/DeepSeek-R1-Distill-Llama-8B, 
      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, 
      deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

复制到箭头位置，保留引号（用v3就不用管）


![image](https://github.com/user-attachments/assets/f1d23be0-c2e8-4a44-be2d-650634c37ccf)




### 6.登录电脑微信，打开程序，对文件传输助手说：泰图斯，你好。若回复，则设置完成。

#### 先登录微信，再打开软件！
#### 打开后ai在操作时，不要动鼠标
#### 不要关闭ai打开的聊天窗口，可以最小化
#### 请注意，呼唤时一定是：xxx（呼唤他的名字），xxx（你的消息）。逗号不可省去


### 6.Config.Json配置说明：

      {
          "DATABASE": {
              "api_key": "api_key_here",	 #你的apikey信息
              "base_url": "https://api.siliconflow.cn/v1/chat/completions",
              "master": "your_name", 	 #你的名字，或者你希望机器人叫你的名字
              "model_set": "deepseek-ai/DeepSeek-V3",
              "prompt_set": "", 		#这里是机器人的角色设定，你可以让他扮演任何角色，只需要写在引号内，我已经预制了一个角色，不懂不要改。
      如果要改记住，文字只能一行，多行会报错
              "called_name": "\u6cf0\u56fe\u65af",		#你呼喊他的名字
              "calling_you_window": "文件传输助手",	#这里是ai自动找你聊天的窗口名称，建议默认设置为文件传输助手，当然你可以有你的小情趣
              "auto_call":0	#是否时不时找你聊天，默认关闭，如果想开启，改为1（活动在早八点到晚上12点）
          }
      }



### 7.group_list.json说明

    []      #加入你想接入机器人的群聊如：”3222胡儿贤父群”。记住用双引号						引用，如果有两个即以上，中间加逗号，记住要用英文逗号和引						号!(, “”)

### 8.listen_list.json 说明

      ["文件传输助手"]  #加入你想接入机器人的聊天个人如："文件传输助手",”xxx”。记住用					双引号引用，如果有两个即以上，中间加逗号，记住要用英文逗					号和引号!(, “”)
当然如果上述listen_list.json 和group_list.json文件设置你觉得麻烦，可以在接入后在微信上发送快捷命令修改，见下


### 9.微信快捷命令：
#### 对任何你接入的窗口都可以使用，但我建议你只对文件传输助手发送信息，不要让其他人知道如何修改你的机器人。

命令如下，效果为字面意思
    "听令，增加人物',
    '听令，增加群组',
    '听令，删除群组',
    '听令，删除人物',
    '听令，列出列表',  #这里是列出listen_list和group_list，这个后面不用写值
    '听令，修改问候窗口', #这里是更改ai自动找你聊天的窗口
    '听令，修改设定'    #这里是修改ai角色预设的地方

使用方式：

'''python
    对文件传输助手发消息：听令，XXXX后面跟你要修改的值
    比如：听令，增加人物张三
    听令，修改问候窗口文件传输助手
    听令，修改设定请你扮演陈桂林
'''
