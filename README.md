# titus-WeChat-robot
# 智能会话助手 - 微信集成解决方案

## 项目概述
本解决方案通过wxauto库实现微信本地机器人，提供拟真对话体验与智能记忆管理。系统采用模块化设计，具备上下文感知、多级记忆存储和动态画像分析等核心功能，有效平衡隐私保护与智能交互需求。

## 核心功能体系

### 1. 交互引擎
- **场景无感调用**：通过特定唤醒词(如"泰图斯")在任意聊天界面触发助手
- **拟真消息模拟**：
  - 自然分段发送
  - 支持表情符号与格式修饰
  - 主动随机聊天（默认关闭，开启见教程）
  - 在代码中已经预设了优质关键词，不会调prompt的小白也能轻松使用

### 2. 记忆管理系统
#### 短期记忆库：
#####ai会记录在该窗口你对ai的聊天记录和对方（群组）对ai的聊天信息。为了隐私安全，ai只会读取你呼唤他的消息记录（如：泰图斯，xxx），不会记录其他日常信息。
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
#### 长期画像分析：
#####ai会在你（或者在聊天框的对方）对他的聊天信息达到8条时，分析你的人物画像。记忆你的性格，特质，兴趣爱好。同时ai会在自己回答12条消息时，总结聊天内容。减少了上下文Token使用。
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
### 3. 自适应交互协议
#### 上下文感知模型
    详见def _generate_reply(self, context, message)函数，可自行diy
###4.微信快捷命令：可以直接在微信上发消息进行修改设置（详见教程），可以自行diy更多命令。命令编写逻辑为：
            ```python
            if content.startswith('听令，'): 为真
                find_XXXX = extract_text_after_keyword(content, '听令，你的命令名字')为真
            if find_XXXX:为真
                # 执行里面的命令逻辑


