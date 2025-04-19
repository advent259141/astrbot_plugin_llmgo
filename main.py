from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp
from PIL import Image, ImageDraw, ImageFont
import io
import json
import os
import numpy as np
import tempfile
from typing import Dict, Any

# 存储活动游戏的状态
# key: event.unified_msg_origin (唯一标识用户/群聊)
# value: 游戏状态字典 (board, user_color, ai_color, current_turn)
active_games: Dict[str, Dict[str, Any]] = {}

@register("llm_go", "Jason.Joestar", "与 LLM 下围棋的插件", "0.1.0", "")
class LLMGoPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 围棋相关常量
        self.board_size = 19
        # 当前目录路径
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        # 预渲染空棋盘底图
        self.base_board_image = self.create_base_board_image()
        # 活动游戏状态存储
        self.active_games = active_games # 使用全局字典

    # --- 指令组定义 ---
    @filter.command_group("llmgo")
    async def llmgo_group(self, event: AstrMessageEvent):
        '''围棋游戏指令组'''
        # 提供基础帮助信息或默认行为
        yield event.plain_result("围棋插件指令：/llmgo start <颜色>, /llmgo move <x> <y>, /llmgo quit")

    @llmgo_group.command("start")
    async def start_game(self, event: AstrMessageEvent, color_choice: str):
        '''开始一局新的围棋游戏。颜色可选：黑棋 或 白棋'''
        user_name = event.get_sender_name()
        game_id = event.unified_msg_origin

        if game_id in self.active_games:
            yield event.plain_result("你已经有一局游戏正在进行中。请先使用 /llmgo quit 结束当前游戏。")
            return

        color_choice = color_choice.strip()
        if color_choice not in ["白棋", "黑棋"]:
            yield event.plain_result("颜色选择无效，请输入 '黑棋' 或 '白棋'")
            return

        # 初始化棋盘和游戏状态
        board = np.full((self.board_size, self.board_size), 2, dtype=int)
        user_color = 0 if color_choice == "黑棋" else 1
        ai_color = 1 - user_color
        current_turn = 0  # 黑棋先手

        game_state = {
            "board": board,
            "user_color": user_color,
            "ai_color": ai_color,
            "current_turn": current_turn
        }
        self.active_games[game_id] = game_state

        # 创建并发送初始棋盘
        board_image = self.render_stones(board)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        board_image.save(temp_file.name)

        initial_message = [
            Comp.Plain(f"你好, {user_name}! 你选择了{color_choice}，游戏开始！\n"),
            Comp.Plain(f"{'你先手' if user_color == 0 else 'AI先手'}。\n"),
            Comp.Plain("使用指令 /llmgo move <x> <y> 来下棋 (例如: /llmgo move 8 10)。\n"),
            Comp.Image(file=temp_file.name)
        ]

        # 如果AI先手，让AI先下
        if current_turn == ai_color:
            ai_move = await self.get_ai_move(board, ai_color, user_color)
            if ai_move:
                x, y = ai_move
                board[x][y] = ai_color
                game_state["current_turn"] = user_color # 轮到用户

                board_image = self.render_stones(board)
                temp_file_ai = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                board_image.save(temp_file_ai.name)

                initial_message.extend([
                    Comp.Plain(f"\nAI 在 ({x},{y}) 落子\n"),
                    Comp.Image(file=temp_file_ai.name)
                ])
            else:
                 initial_message.append(Comp.Plain("\nAI未能下出第一步棋，请你先下。"))
                 game_state["current_turn"] = user_color # 轮到用户

        message_result = event.make_result()
        message_result.chain = initial_message
        await event.send(message_result)

    @llmgo_group.command("move")
    async def make_move(self, event: AstrMessageEvent, x: int, y: int):
        '''在指定坐标 (x, y) 落子'''
        game_id = event.unified_msg_origin

        if game_id not in self.active_games:
            yield event.plain_result("你还没有开始游戏。请使用 /llmgo start <颜色> 开始新游戏。")
            return

        game_state = self.active_games[game_id]
        board = game_state["board"]
        user_color = game_state["user_color"]
        ai_color = game_state["ai_color"]
        current_turn = game_state["current_turn"]

        if current_turn != user_color:
            yield event.plain_result("现在不是你的回合。")
            return

        # 验证坐标
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            yield event.plain_result(f"坐标 ({x},{y}) 超出范围 (0-18)。")
            return

        if board[x][y] != 2:
            yield event.plain_result(f"位置 ({x},{y}) 已有棋子。")
            return

        # 用户落子
        board[x][y] = user_color
        game_state["current_turn"] = ai_color # 轮到AI

        board_image_user = self.render_stones(board)
        temp_file_user = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        board_image_user.save(temp_file_user.name)

        user_move_message = event.make_result()
        user_move_message.chain = [
            Comp.Plain(f"你在 ({x},{y}) 落子\n"),
            Comp.Image(file=temp_file_user.name)
        ]
        await event.send(user_move_message)

        # AI回合
        ai_move = await self.get_ai_move(board, ai_color, user_color)
        if ai_move:
            ai_x, ai_y = ai_move
            board[ai_x][ai_y] = ai_color
            game_state["current_turn"] = user_color # 轮到用户

            board_image_ai = self.render_stones(board)
            temp_file_ai = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            board_image_ai.save(temp_file_ai.name)

            ai_move_message = event.make_result()
            ai_move_message.chain = [
                Comp.Plain(f"AI 在 ({ai_x},{ai_y}) 落子\n"),
                Comp.Image(file=temp_file_ai.name)
            ]
            await event.send(ai_move_message)
        else:
            # AI未能落子，可能棋盘已满或出错
            game_state["current_turn"] = user_color # 轮到用户继续
            yield event.plain_result("AI 未能落子，轮到你了。")


    @llmgo_group.command("quit")
    async def quit_game(self, event: AstrMessageEvent):
        '''结束当前的围棋游戏'''
        game_id = event.unified_msg_origin

        if game_id in self.active_games:
            del self.active_games[game_id]
            yield event.plain_result("游戏已结束。")
        else:
            yield event.plain_result("你没有正在进行的游戏。")

    def create_base_board_image(self):
        """创建基础围棋棋盘图像（无棋子）"""
        cell_size = 30
        margin = 20
        board_size = self.board_size
        
        # 计算图像尺寸
        img_size = cell_size * (board_size - 1) + 2 * margin
        
        # 创建图像
        img = Image.new("RGB", (img_size, img_size), (240, 200, 150))
        draw = ImageDraw.Draw(img)
        
        # 画网格线
        for i in range(board_size):
            # 横线
            draw.line([(margin, margin + i * cell_size), 
                      (img_size - margin, margin + i * cell_size)], fill=(0, 0, 0), width=1)
            # 竖线
            draw.line([(margin + i * cell_size, margin), 
                      (margin + i * cell_size, img_size - margin)], fill=(0, 0, 0), width=1)
        
        # 标记天元和星位
        star_points = [3, 9, 15] if board_size == 19 else [3, board_size // 2, board_size - 4]
        for x in star_points:
            for y in star_points:
                draw.ellipse([(margin + x * cell_size - 3, margin + y * cell_size - 3),
                             (margin + x * cell_size + 3, margin + y * cell_size + 3)], fill=(0, 0, 0))
                
        # 添加坐标标签
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
            
        for i in range(board_size):
            # 顶部数字坐标
            draw.text((margin + i * cell_size, 5), str(i), fill=(0, 0, 0), font=font)
            # 左侧数字坐标
            draw.text((5, margin + i * cell_size - 6), str(i), fill=(0, 0, 0), font=font)
        
        return img

    def render_stones(self, board):
        """仅渲染棋子，在基础棋盘上"""
        # 复制基础棋盘
        img = self.base_board_image.copy()
        draw = ImageDraw.Draw(img)
        
        cell_size = 30
        margin = 20
        
        # 绘制棋子
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == 0:  # 黑棋
                    draw.ellipse([(margin + x * cell_size - 12, margin + y * cell_size - 12),
                                 (margin + x * cell_size + 12, margin + y * cell_size + 12)], fill=(0, 0, 0))
                elif board[x][y] == 1:  # 白棋
                    draw.ellipse([(margin + x * cell_size - 12, margin + y * cell_size - 12),
                                 (margin + x * cell_size + 12, margin + y * cell_size + 12)], fill=(255, 255, 255), outline=(0, 0, 0))
        
        return img

    async def get_ai_move(self, board, ai_color, user_color):
        """让LLM决定下一步棋"""
        func_tools_mgr = self.context.get_llm_tool_manager()
        
        # 准备系统提示
        color_name = "黑棋" if ai_color == 0 else "白棋"
        opponent_color_name = "白棋" if ai_color == 0 else "黑棋"
        
        # 把棋盘状态转换为JSON字符串
        board_json = json.dumps(board.tolist())
        
        system_prompt = f"""
        你是一个围棋AI助手，你执{color_name}，用户执{opponent_color_name}。
        以下是当前棋盘状态的JSON格式（19x19矩阵）：
        {board_json}
        
        其中：
        - 0 表示黑棋
        - 1 表示白棋
        - 2 表示空位置
        
        请分析当前棋局并给出你的下一步落子位置。
        你必须严格按照以下格式回复坐标：X,Y
        其中X和Y是0到18之间的整数，表示落子位置的坐标。
        
        确保你选择的位置是空的（值为2），不要在已有棋子的位置落子。
        不要输出任何其他内容，只回复坐标，例如：9,10
        """
        
        try:
            llm_response = await self.context.get_using_provider().text_chat(
                prompt="我正在等待你下一步棋，请根据上述棋盘状态，选择一个合理的位置落子。",
                contexts=[{"role": "system", "content": system_prompt}],
                image_urls=[],
                func_tool=func_tools_mgr
            )
            
            if llm_response.role == "assistant":
                move_str = llm_response.completion_text.strip()
                try:
                    x, y = map(int, move_str.split(','))
                    # 验证坐标有效
                    if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x][y] == 2:
                        return (x, y)
                    else:
                        # 随机生成一个有效坐标（作为备选）
                        empty_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if board[i][j] == 2]
                        if empty_positions:
                            import random
                            return random.choice(empty_positions)
                except Exception:
                    pass
            
            # 如果无法获得有效坐标，返回None
            return None
            
        except Exception:
            return None

    async def terminate(self):
        '''插件卸载/停用时调用'''
        # 清理可能残留的游戏状态
        self.active_games.clear()
        pass
