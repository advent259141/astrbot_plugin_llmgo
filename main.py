from astrbot.api.all import *
from astrbot.api.event.filter import *
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import tempfile
from typing import Dict, Any, List, Tuple, Set

# 存储活动游戏的状态
# key: event.unified_msg_origin (唯一标识用户/群聊)
# value: 游戏状态字典 (board, user_color, ai_color, current_turn)
active_games: Dict[str, Dict[str, Any]] = {}

@register("llm_go", "GitHub Copilot", "与 LLM 下围棋的插件", "0.1.0", "https://github.com/advent259141/astrbot_plugin_llmgo")
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
    @command_group("llmgo")
    async def llmgo_group(self, event: AstrMessageEvent):
        '''围棋游戏指令组'''
        # 提供基础帮助信息或默认行为
        yield event.plain_result("围棋插件指令：/llmgo start <颜色>, /llmgo place <x> <y>, /llmgo quit")

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
            Plain(f"你好, {user_name}! 你选择了{color_choice}，游戏开始！\n"),
            Plain(f"{'你先手' if user_color == 0 else 'AI先手'}。\n"),
            Plain("使用指令 /llmgo place <x> <y> 来下棋 (例如: /llmgo place 8 10)。\n"),
            Image(file=temp_file.name)
        ]
        yield event.chain_result(initial_message)

        # 如果AI先手，让AI先下
        if current_turn == ai_color:
            # AI第一步不需要考虑吃子
            ai_place = await self.get_ai_place(board, ai_color)
            if ai_place:
                x, y = ai_place
                board[x][y] = ai_color
                game_state["current_turn"] = user_color # 轮到用户

                board_image = self.render_stones(board)
                temp_file_ai = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                board_image.save(temp_file_ai.name)

                message = [
                    Plain(f"AI 在 ({x},{y}) 落子\n"),
                    Image(file=temp_file_ai.name)
                ]
                yield event.chain_result(message)
            else:
                 yield event.plain_result("AI未能下出第一步棋，请你先下。")
                 game_state["current_turn"] = user_color # 轮到用户

    @llmgo_group.command("place")
    async def place(self, event: AstrMessageEvent, x: int, y: int):
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

        # --- 用户回合 ---
        # 1. 用户落子 (先尝试落子)
        board[x][y] = user_color

        # 2. 检查用户是否吃掉了AI的子
        captured_by_user = self.find_and_replace_captured(board, x, y, ai_color)
        capture_message = f"你吃掉了{len(captured_by_user)}颗棋子！\n" if captured_by_user else ""

        # 4. 发送用户落子和吃子结果
        board_image_user = self.render_stones(board)
        temp_file_user = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        board_image_user.save(temp_file_user.name)
        user_place_message = [
            Plain(f"你在 ({x},{y}) 落子\n{capture_message}"),
            Image(file=temp_file_user.name)
        ]
        yield event.chain_result(user_place_message)

        # 5. 切换到AI回合
        game_state["current_turn"] = ai_color

        # --- AI 回合 ---
        # 1. 获取AI落子位置
        ai_place = await self.get_ai_place(board, ai_color)
        if ai_place:
            ai_x, ai_y = ai_place

            # 2. AI落子
            board[ai_x][ai_y] = ai_color

            # 3. 检查AI是否吃掉了用户的子
            captured_by_ai = self.find_and_replace_captured(board, ai_x, ai_y, user_color)
            ai_capture_message = f"AI吃掉了{len(captured_by_ai)}颗棋子！\n" if captured_by_ai else ""

            # 5. 发送AI落子和吃子结果
            board_image_ai = self.render_stones(board)
            temp_file_ai = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            board_image_ai.save(temp_file_ai.name)
            ai_place_message = [
                Plain(f"AI 在 ({ai_x},{ai_y}) 落子\n{ai_capture_message}"),
                Image(file=temp_file_ai.name)
            ]
            yield event.chain_result(ai_place_message)

            # 6. 切换回用户回合
            game_state["current_turn"] = user_color
        else:
            # AI未能落子
            game_state["current_turn"] = user_color # 轮到用户继续
            yield event.plain_result("AI 未能落子，轮到你了。")

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取一个点的相邻点坐标"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    def get_group_liberties(self, board: np.ndarray, x: int, y: int, color: int) -> Tuple[int, Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        计算包含(x, y)的棋子组的气数、组成员和气的位置。
        返回: (气数, 组成员坐标集合, 气坐标集合)
        """
        if board[x][y] != color:
            return 0, set(), set()

        q = [(x, y)]
        visited_stones = {(x, y)}
        liberties = set()

        while q:
            cx, cy = q.pop(0)
            for nx, ny in self.get_neighbors(cx, cy):
                neighbor_state = board[nx][ny]
                if neighbor_state == 2: # 空点是气
                    liberties.add((nx, ny))
                elif neighbor_state == color and (nx, ny) not in visited_stones:
                    visited_stones.add((nx, ny))
                    q.append((nx, ny))
        return len(liberties), visited_stones, liberties

    def find_and_replace_captured(self, board: np.ndarray, place_x: int, place_y: int, opponent_color: int) -> List[Tuple[int, int]]:
        """
        在落子(place_x, place_y)后，查找并移除被吃掉的对方棋子。
        返回被吃掉的棋子坐标列表。
        """
        captured_stones_total = []
        for nx, ny in self.get_neighbors(place_x, place_y):
            if board[nx][ny] == opponent_color:
                liberty_count, group_stones, _ = self.get_group_liberties(board, nx, ny, opponent_color)
                if liberty_count == 0:
                    # 这个组被吃掉了
                    for gx, gy in group_stones:
                        board[gx][gy] = 2 # 从棋盘上移除
                        captured_stones_total.append((gx, gy))
        return captured_stones_total

    async def get_ai_place(self, board, ai_color):
        """让LLM决定下一步棋 (不再处理吃子判断)"""
        global json
        func_tools_mgr = self.context.get_llm_tool_manager()

        # 准备系统提示
        color_name = "黑棋" if ai_color == 0 else "白棋"
        opponent_color_name = "白棋" if ai_color == 0 else "黑棋"

        # 只提取有棋子的位置
        black_stones = []
        white_stones = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] == 0:
                    black_stones.append([r, c])
                elif board[r][c] == 1:
                    white_stones.append([r, c])

        system_prompt = f"""
        你是一个围棋AI助手，你执{color_name}，用户执{opponent_color_name}。
        当前棋盘状态：
        - 黑棋位置: {black_stones}
        - 白棋位置: {white_stones}

        棋盘大小是19x19，坐标范围是0-18。

        围棋规则和目标：
        1.  **目标**: 最终目标是比对手围住更多的空白交叉点（称为“地”或“目”）。同时，也要尽可能多地吃掉对方的棋子。
        2.  **落子**: 轮流在棋盘的空白交叉点上放置棋子。
        3.  **气**: 一个棋子或一组相连的同色棋子，其上下左右直接相邻的空白交叉点称为“气”。
        4.  **吃子**: 如果一个棋子或一组相连的棋子所有的“气”都被对方棋子占据，它们就会被从棋盘上提走（吃掉）。
        5.  **禁入点 (自杀)**: 通常情况下，不能在没有“气”的位置落子，除非这个落子能立刻吃掉对方的棋子（即填掉对方棋子组的最后一口气）。
        6.  **活棋**: 为了让自己的棋子不被吃掉，需要确保它们至少有两个或更多分开的“眼”（由己方棋子围住的内部空点）。只有拥有两个真眼的棋块是绝对活棋。
        7.  **打劫**: 这是一个复杂的规则，用于防止双方无限重复提子。简单来说，如果一方提走对方一个子，对方不能立即在同一位置提回，必须先在别处下一手。 (你可以暂时简化或忽略打劫的复杂计算，但要避免明显重复的提子局面)。

        你的任务：
        - 分析当前棋局，考虑攻防、围地、棋子死活等因素。
        - 选择一个符合规则且对你最有利的落子位置。
        - 优先考虑能吃掉对方重要棋子、扩大自己地盘、确保自己棋子存活或威胁对方棋子的位置。
        - 避免下在明显会被吃掉或没有意义的位置。
        - **绝对禁止自杀行为** (除非该落子能吃掉对方棋子)。

        请给出你的下一步落子位置。
        你必须严格按照以下格式回复坐标：X,Y
        其中X和Y是0到18之间的整数，表示落子位置的坐标。

        确保你选择的位置没有棋子。
        不要输出任何其他内容，只回复坐标，例如：9,10
        """

        try:
            llm_response = await self.context.get_using_provider().text_chat(
                prompt="请根据围棋规则和当前局面，选择你的最佳落子位置。",
                contexts=[{"role": "system", "content": system_prompt}],
                image_urls=[],
                func_tool=func_tools_mgr
            )

            if llm_response.role == "assistant":
                place_str = llm_response.completion_text.strip()
                try:
                    # 尝试解析 JSON 或 纯坐标
                    try:
                        import json
                        data = json.loads(place_str)
                        if isinstance(data, dict) and "place" in data and isinstance(data["place"], list) and len(data["place"]) == 2:
                            x, y = data["place"]
                        elif isinstance(data, list) and len(data) == 2:
                             x, y = data
                        else:
                             raise ValueError("Invalid format")
                    except (json.JSONDecodeError, ValueError):
                         x, y = map(int, place_str.split(','))

                    # 验证坐标有效
                    if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x][y] == 2:
                        return x, y
                    else:
                        pass
                except Exception:
                    pass

            # 如果无法获得有效坐标，返回随机位置
            empty_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if board[i][j] == 2]
            if empty_positions:
                import random
                return random.choice(empty_positions)

            return None

        except Exception:
            return None

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

    async def terminate(self):
        '''插件卸载/停用时调用'''
        # 清理可能残留的游戏状态
        self.active_games.clear()
        pass

    @llm_tool("start_go_game")
    async def start_go(self, event: AstrMessageEvent, color_choice: str):
        """开始一局新的围棋游戏

        Args:
            color_choice (string): `黑棋` 或者 `白旗`
        """
        async for result in self.start_game(event, color_choice):
            yield result

    @llm_tool("place_stone")
    async def place_stone(self, event: AstrMessageEvent, x: int, y: int):
        """围棋游戏中，下一颗棋子

        Args:
            x (int): 棋子横坐标，应在 0 至 18 之间
            y (int): 棋子纵坐标，应在 0 至 18 之间
        """
        async for result in self.place(event, x, y):
            yield result

    @llm_tool("quit_go_game")
    async def quit_go(self, event: AstrMessageEvent):
        """退出围棋游戏

        Args:

        """
        async for result in self.quit_game(event):
            yield result