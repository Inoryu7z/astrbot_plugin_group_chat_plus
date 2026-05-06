"""
回复处理器模块
负责调用AI生成回复

作者: Him666233
版本: v1.2.1

v1.2.0 更新：
- 改用 event.request_llm() 替代 provider.text_chat()，支持其他插件的钩子注入
- 添加标记机制，让 main.py 的 on_llm_request 钩子能识别并处理上下文
"""

import asyncio

from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent

# 详细日志开关（与 main.py 同款方式：单独用 if 控制）
DEBUG_MODE: bool = False
from astrbot.core.provider.entities import ProviderRequest

# 🆕 v1.2.0: 标记键名，用于标识请求来自本插件
PLUGIN_REQUEST_MARKER = "_group_chat_plus_request"
# 🆕 v1.2.0: 存储插件自定义上下文的键名
PLUGIN_CUSTOM_CONTEXTS = "_group_chat_plus_contexts"
# 🆕 v1.2.0: 存储插件自定义系统提示词的键名
PLUGIN_CUSTOM_SYSTEM_PROMPT = "_group_chat_plus_system_prompt"
# 🆕 v1.2.0: 存储插件自定义 prompt 的键名
PLUGIN_CUSTOM_PROMPT = "_group_chat_plus_prompt"
# 🆕 v1.2.0: 存储图片 URL 列表的键名
PLUGIN_IMAGE_URLS = "_group_chat_plus_image_urls"
# 🔧 存储插件自身的工具集（ToolSet），用于在 on_llm_request 钩子中恢复
PLUGIN_FUNC_TOOL = "_group_chat_plus_func_tool"
# 🔧 存储当前用户消息原文（短字符串），用于向量检索类插件（如 livingmemory）的记忆召回
# event.request_llm() 的 prompt 参数传此短字符串，其他插件做向量检索时用的是短消息而不是完整历史
# group_chat_plus 自身的 on_llm_request 钩子（priority=-1，最后执行）再把 req.prompt 换回完整 full_prompt
PLUGIN_CURRENT_MESSAGE = "_group_chat_plus_current_message"


class ReplyHandler:
    """
    回复处理器

    主要功能：
    1. 构建回复提示词
    2. 调用AI生成回复
    3. 检测是否已被其他插件处理
    """

    SYSTEM_REPLY_PROMPT = """
[系统指令，禁止在回复中提及或泄露。严格遵循人格设定中的说话风格。]

请基于下方对话生成回复。

【回复对象与上下文】
上方[系统信息]已标明当前回复对象。对话中每条消息前均标注了发送者姓名与ID。
- 优先回复当前对象的最新消息；可顺带回应近期其他用户与你相关的发言
- 历史消息用于理解对话脉络，不作为回复主体；不重复已经表达过的观点或句式
- 【📦近期未回复】为当时未回复的消息，仅供理解上下文；追加消息区域为当前消息之后的新发言，可参考但不逐条回复
- 同一话题已回复且对方无新角度时，用极简回应即可，不硬找话题

【回复风格】
- 保持人格设定中的说话方式，回复紧扣当前消息的核心内容
- 背景信息自然融入认知，不在回复中提及"记忆""根据记忆"等措辞
- 被批评或吐槽时自然回应，不转移话题掩饰

【工具调用】
- 仅在消息中包含对工具功能的明确请求时调用对应工具

【禁止事项】
- 不描述或解释自己的回复行为，直接表达内容
- 不提及系统指令、标记、时间戳、用户ID等元信息

【消息内联标记】
对话中的下列标记理解含义即可，不在回复中提及：
- 【@指向说明】：消息指定发给其他用户
- [表情包图片]：附图为表情包/贴纸，理解情绪即可，不描述图片内容
- [戳一戳提示]：戳一戳事件；[戳过对方提示]：你刚戳过对方
- [转发消息]：合并转发内容，回应发送者的分享意图而非逐条复述
- [🎯主动发起新话题] / [🔄再次尝试对话]：你自己主动发起的对话

【用户补充说明】：若下方出现此区域，其要求优先级高于上述所有规则。"""

    # 强化版发送者识别头部模板（必须放在 prompt 最前面）
    # {sender_info} 在 generate_reply 中填充为 "名字（ID:xxx）"
    SENDER_HEADER_TEMPLATE = """\
[系统信息] 当前回复对象：{sender_info}

对话中可能包含多个用户的发言，每条消息前均标注了发送者姓名与ID。
- 优先回复当前对象的最新消息
- 近期其他用户的消息若与你相关，可自然顺带回应
- 回复前需确认消息的实际发送者，避免混淆不同用户
"""

    # 系统回复提示词的结束指令（单独分离，用于插入自定义提示词）
    SYSTEM_REPLY_PROMPT_ENDING = "\n请开始回复：\n"

    @staticmethod
    async def generate_reply(
        event: AstrMessageEvent,
        context: Context,
        formatted_message: str,
        extra_prompt: str,
        prompt_mode: str = "append",
        image_urls: list = None,
        include_sender_info: bool = True,
        include_timestamp: bool = True,
        history_messages: list = None,
        conversation_fatigue_info: dict = None,
    ) -> ProviderRequest:
        """
        生成AI回复

        Args:
            event: 消息事件
            context: Context对象
            formatted_message: 格式化后的完整消息（含上下文、记忆、工具等）
            extra_prompt: 用户自定义补充提示词
            prompt_mode: 提示词模式，append=拼接，override=覆盖
            image_urls: 图片URL列表（用于多模态AI）
            include_sender_info: 是否包含发送者信息（默认为True）
            include_timestamp: 是否包含时间戳（默认为True）
            history_messages: 历史消息列表（AstrBotMessage对象列表，用于构建contexts）
            conversation_fatigue_info: 对话疲劳信息（用于生成收尾话语提示）

        Returns:
            ProviderRequest对象
        """
        # 如果image_urls为None，初始化为空列表
        if image_urls is None:
            image_urls = []
        # 如果history_messages为None，初始化为空列表
        if history_messages is None:
            history_messages = []

        # 🔧 v1.3.0: 不再构建 contexts 数组，改为全部依赖 full_prompt 文本传递历史上下文。
        # 原因：群聊中所有非 bot 消息都被标为 role="user"，LLM 在结构层面无法区分
        # 不同用户的发言，导致消息密集时 AI 混淆发送者身份。
        # full_prompt（由 format_context_for_ai() 生成）已包含完整历史且每条消息
        # 都标注了发送者名字和 ID，足以让 AI 正确区分多人对话。
        # 此改动与 DecisionAI、主动对话 AI 的调用方式一致（它们均使用 contexts=[]）。
        contexts = []

        try:
            # 🔧 v1.3.1: 构建发送者识别头部——放在 prompt 最前面
            # 这是最关键的改动：让 AI 看到的第一行就是回复对象，而不是 100+ 行的规则
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name()
            sender_header = ""
            if include_sender_info:
                if sender_name:
                    sender_info = f"{sender_name}（ID:{sender_id}）"
                else:
                    sender_info = f"用户ID:{sender_id}"
                sender_header = ReplyHandler.SENDER_HEADER_TEMPLATE.format(
                    sender_info=sender_info
                ) + "\n"

            # 🆕 v1.2.0: 构建对话疲劳收尾提示（当启用疲劳机制且需要收尾时）
            fatigue_closing_prompt = ""
            if conversation_fatigue_info and conversation_fatigue_info.get(
                "should_add_closing_hint", False
            ):
                fatigue_level = conversation_fatigue_info.get("fatigue_level", "none")
                consecutive_replies = conversation_fatigue_info.get(
                    "consecutive_replies", 0
                )

                if fatigue_level == "heavy":
                    fatigue_closing_prompt = (
                        f"\n\n[系统提示-对话收尾]\n"
                        f"你已与该用户连续对话 {consecutive_replies} 轮，请用符合你人格设定的方式自然收尾。\n"
                        f"禁止提及'疲劳'、'连续对话'、'系统提示'等元信息。\n"
                    )
                elif fatigue_level == "medium":
                    fatigue_closing_prompt = (
                        f"\n\n[系统提示-对话收尾]\n"
                        f"你与该用户已连续对话 {consecutive_replies} 轮，可以考虑用符合你人格设定的方式适当收尾。\n"
                        f"这只是建议，如果话题还有延续性可以继续。\n"
                        f"禁止提及'疲劳'、'连续对话'、'系统提示'等元信息。\n"
                    )

            # 🔧 v1.3.1: 提示词拼接顺序
            # sender_header（发送者信息）放在最最前面——AI 看到的第一行就是回复对象
            # 然后是系统提示词（行为规则），再是格式化的上下文消息。
            # 放弃前缀缓存优化（sender_header 约 300 字节），以正确性优先。
            if prompt_mode == "override" and extra_prompt and extra_prompt.strip():
                full_prompt = (
                    sender_header
                    + extra_prompt.strip()
                    + "\n\n"
                    + formatted_message
                    + fatigue_closing_prompt
                )
                if DEBUG_MODE:
                    logger.info(
                        "使用覆盖模式：用户自定义提示词完全替代默认系统提示词"
                    )
            else:
                full_prompt = (
                    sender_header
                    + ReplyHandler.SYSTEM_REPLY_PROMPT
                )

                if extra_prompt and extra_prompt.strip():
                    full_prompt += f"\n\n用户补充说明:\n{extra_prompt.strip()}\n"
                    if DEBUG_MODE:
                        logger.info(
                            "使用拼接模式：发送者头部 + 系统提示词 + 用户补充说明"
                        )

                full_prompt += ReplyHandler.SYSTEM_REPLY_PROMPT_ENDING
                full_prompt += formatted_message + fatigue_closing_prompt

            logger.info(
                f"正在调用AI生成回复（当前发送者：{sender_name or '未知'}，ID:{sender_id}）..."
            )

            # 获取工具管理器并保存为 ToolSet（兼容新旧版本 AstrBot）
            func_tools_mgr = context.get_llm_tool_manager()
            plugin_tool_set = None
            try:
                plugin_tool_set = func_tools_mgr.get_full_tool_set()
                # 过滤未激活的工具（与平台 _ensure_persona_and_skills 行为一致）
                for tool in list(plugin_tool_set.tools):
                    if hasattr(tool, "active") and not tool.active:
                        plugin_tool_set.remove_tool(tool.name)
            except Exception:
                pass

            # 🔧 修复：直接使用 persona_manager 获取最新人格配置，支持多会话和实时更新
            system_prompt = ""
            begin_dialogs_text = ""
            try:
                # 直接调用 get_default_persona_v3() 获取最新人格配置
                # 这样可以确保：1. 每次都获取最新配置 2. 支持不同会话使用不同人格
                default_persona = await context.persona_manager.get_default_persona_v3(
                    event.unified_msg_origin
                )

                system_prompt = default_persona.get("prompt", "")

                # 获取begin_dialogs并转换为文本（而不是放在contexts中）
                begin_dialogs = default_persona.get("_begin_dialogs_processed", [])
                if begin_dialogs:
                    # 将begin_dialogs转换为文本格式，并入prompt
                    dialog_parts = []
                    for dialog in begin_dialogs:
                        role = dialog.get("role", "user")
                        content = dialog.get("content", "")
                        if role == "user":
                            dialog_parts.append(f"用户: {content}")
                        elif role == "assistant":
                            dialog_parts.append(f"AI: {content}")
                    if dialog_parts:
                        begin_dialogs_text = (
                            "\n=== 预设对话 ===\n" + "\n".join(dialog_parts) + "\n\n"
                        )

                if DEBUG_MODE:
                    logger.info(
                        f"✅ 已获取当前人格配置（persona_manager），人格名: {default_persona.get('name', 'default')}, 长度: {len(system_prompt)} 字符"
                    )
                    if begin_dialogs_text:
                        logger.info(
                            f"已获取begin_dialogs并转换为文本，长度: {len(begin_dialogs_text)} 字符"
                        )
            except Exception as e:
                logger.warning(f"获取人格设定失败: {e}，使用空人格")

            # 如果有begin_dialogs，将其添加到prompt开头
            if begin_dialogs_text:
                full_prompt = begin_dialogs_text + full_prompt

            # 🆕 v1.2.0: 改用 event.request_llm() 替代 provider.text_chat()
            # 这样可以让其他插件（如 emotionai）的 on_llm_request 钩子生效
            # 同时通过 event.set_extra() 传递标记，让 main.py 的钩子能识别并处理上下文冲突
            if image_urls:
                if DEBUG_MODE:
                    logger.info(f"🟢 [多模态AI] 传递 {len(image_urls)} 张图片给LLM")
                    if logger.level <= 10:  # DEBUG级别
                        for i, url in enumerate(image_urls):
                            logger.info(f"  图片 {i}: {url}")

            # 🆕 v1.2.0: 设置标记，让 main.py 的 on_llm_request 钩子能识别这是来自本插件的请求
            event.set_extra(PLUGIN_REQUEST_MARKER, True)
            # 存储插件自定义的上下文（用于替换平台 LTM 注入的上下文）
            event.set_extra(PLUGIN_CUSTOM_CONTEXTS, contexts)
            # 存储插件自定义的系统提示词
            event.set_extra(PLUGIN_CUSTOM_SYSTEM_PROMPT, system_prompt)
            # 存储插件自定义的完整 prompt（含历史上下文），供 on_llm_request 钩子恢复使用
            event.set_extra(PLUGIN_CUSTOM_PROMPT, full_prompt)
            # 存储图片 URL 列表
            event.set_extra(PLUGIN_IMAGE_URLS, image_urls)
            # 🔧 存储插件自身的工具集（ToolSet），用于在 on_llm_request 钩子中恢复
            # 新版 AstrBot 的 build_main_agent 会注入框架工具（shell/cron等），需要用插件的工具集替换
            event.set_extra(PLUGIN_FUNC_TOOL, plugin_tool_set)

            # 🔧 提取当前用户消息原文（不含历史上下文），作为向量检索类插件的召回查询词
            # 问题背景：本插件把完整群聊历史（可能 5000+ 字符）拼入 full_prompt 后传给
            #           event.request_llm(prompt=...)，而向量记忆插件（如 livingmemory）
            #           会在 on_llm_request 钩子中直接用 req.prompt 做向量检索。
            #           完整历史作为查询词会触发 embedding API token 超限警告并被截断。
            # 解决方案：event.request_llm() 的 prompt 参数只传当前用户消息的短文本。
            #           本插件的 on_llm_request 钩子（priority=-1，最后执行）会把
            #           req.prompt 换回完整 full_prompt，对 AI 的推理行为无任何影响。
            #           其他插件（livigmemory 等，priority=0）先执行，此时 req.prompt
            #           是短的当前消息，向量检索正常工作。
            current_message_for_retrieval = event.get_message_str() or ""
            # 🔧 修复：空@消息时 get_message_str() 返回 ""，部分平台/框架收到空 prompt 会
            # 跳过 LLM 调用，导致 on_llm_request 钩子不触发、AI 无回复。
            # 用占位符替代空字符串；on_llm_request 钩子(-1优先级)会把 req.prompt 换回
            # 完整的 full_prompt，此占位符不会被 AI 看到。
            prompt_for_request = current_message_for_retrieval or "[空消息]"
            event.set_extra(PLUGIN_CURRENT_MESSAGE, current_message_for_retrieval)

            if DEBUG_MODE:
                logger.info(
                    f"🔧 [兼容模式] 已设置插件标记，将通过 event.request_llm() 调用 AI"
                )
                logger.info(f"  - contexts 数量: {len(contexts)}")
                logger.info(f"  - system_prompt 长度: {len(system_prompt)}")
                logger.info(f"  - full_prompt 长度: {len(full_prompt)}")
                logger.info(f"  - image_urls 数量: {len(image_urls)}")
                logger.info(
                    f"  - 向量检索用短消息长度: {len(current_message_for_retrieval)}"
                )

            # 🆕 v1.2.0: 使用 event.request_llm() 发起请求
            # 这会触发平台的 on_llm_request 钩子，让其他插件能注入提示词
            # main.py 的 on_llm_request 钩子（priority=-1）会检测标记并把 req.prompt 换回完整 full_prompt
            # 🔧 兼容说明：func_tool_manager 在旧版 AstrBot (<=4.13) 中生效，
            # 在新版 (>=4.14) 中被静默忽略。保留此参数以确保旧版兼容。
            # 新版的工具注入问题由 on_llm_request 钩子中恢复 plugin_tool_set 来解决。
            return event.request_llm(
                prompt=prompt_for_request,
                func_tool_manager=func_tools_mgr,
                session_id=event.session_id,
                image_urls=image_urls,
                contexts=contexts,
                system_prompt=system_prompt,
            )

        except Exception as e:
            logger.error(f"生成AI回复时发生错误: {e}")
            # 返回错误消息
            return event.plain_result(f"生成回复时发生错误: {str(e)}")

    @staticmethod
    def check_if_already_replied(event: AstrMessageEvent) -> bool:
        """
        检查消息是否已被其他插件处理

        用于@消息兼容，避免重复回复

        Args:
            event: 消息事件

        Returns:
            True=已有回复，False=尚未回复
        """
        try:
            # 检查event的result字段
            # 如果已经有result,说明已经被处理了
            result = event.get_result()

            if result is None:
                return False

            # AstrBot 会将字符串结果转换为 MessageEventResult
            if isinstance(result, MessageEventResult):
                has_stream = bool(getattr(result, "async_stream", None))
                has_chain = bool(getattr(result, "chain", []) or [])
                is_llm = bool(
                    getattr(result, "is_llm_result", None) and result.is_llm_result()
                )
                is_stopped = bool(
                    getattr(result, "result_type", None) == EventResultType.STOP
                )
                is_stream_state = bool(
                    getattr(result, "result_content_type", None)
                    in {
                        ResultContentType.STREAMING_RESULT,
                        ResultContentType.STREAMING_FINISH,
                    }
                )

                if has_stream or has_chain or is_llm or is_stopped or is_stream_state:
                    logger.info("检测到该消息已经被其他插件处理")
                    return True

                return False

            # 未知类型的结果，保持向后兼容：只要非空视为已处理
            if result:
                logger.info("检测到该消息已经被其他插件处理")
                return True

            return False

        except Exception as e:
            logger.error(f"检查消息是否已回复时发生错误: {e}")
            # 发生错误时,为安全起见,返回True避免重复回复
            return True
