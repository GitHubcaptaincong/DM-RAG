import re
import time
import random

from main import talk_with_llm

class DMDemo:
    def __init__(self, script_name):
        """
        初始化剧本主持人

        Args:
            script_name: 剧本名称，用于加载相关剧本信息
        """
        self.script_name = script_name
        self.players = {}  # 存储玩家信息 {player_id: {"name": name, "role": role, ...}}
        self.current_phase = "intro"  # 游戏当前阶段
        self.game_state = {
            "revealed_clues": [],
            "current_scene": "开场",
            "completed_interactions": [],
            "game_progress": 0  # 0-100的进度
        }

        # 加载剧本基本信息
        self.script_info = self._load_script_info()

    def _load_script_info(self):
        """加载剧本基本信息"""
        # 这里可以从数据库或文件加载，现在用talk_with_llm模拟
        basic_info = talk_with_llm(f"请提供《{self.script_name}》剧本的基本信息，包括背景故事、角色列表和主要场景")
        return {
            "title": self.script_name,
            "info": basic_info,
            "roles": self._extract_roles(basic_info)
        }

    def _extract_roles(self, script_info):
        """从剧本信息中提取角色列表"""
        # 这里使用talk_with_llm提取角色信息
        roles_info = talk_with_llm(f"请列出《{self.script_name}》中所有可供玩家扮演的角色名称及其简短描述")
        # 解析返回的角色信息为结构化数据
        roles = {}
        # 简单解析示例，实际应根据LLM返回格式调整
        for line in roles_info.split('\n'):
            if ':' in line:
                name, desc = line.split(':', 1)
                roles[name.strip()] = {"description": desc.strip()}
        return roles

    def assign_roles(self, player_ids, role_preferences=None):
        """
        为玩家分配角色

        Args:
            player_ids: 玩家ID列表
            role_preferences: 可选，玩家角色偏好 {player_id: [preferred_roles]}
        """
        available_roles = list(self.script_info["roles"].keys())
        if len(player_ids) > len(available_roles):
            raise ValueError(f"玩家数量({len(player_ids)})超过可用角色数量({len(available_roles)})")

        # 处理玩家偏好
        assigned_roles = {}
        if role_preferences:
            # 先处理有偏好的玩家
            for player_id, preferences in role_preferences.items():
                for role in preferences:
                    if role in available_roles:
                        assigned_roles[player_id] = role
                        available_roles.remove(role)
                        break

        # 随机分配剩余角色
        unassigned_players = [pid for pid in player_ids if pid not in assigned_roles]
        random.shuffle(available_roles)
        for i, player_id in enumerate(unassigned_players):
            if i < len(available_roles):
                assigned_roles[player_id] = available_roles[i]

        # 更新玩家信息
        for player_id, role in assigned_roles.items():
            self.players[player_id] = {
                "role": role,
                "clues": [],
                "knowledge": []
            }

        return assigned_roles

    def start_game(self):
        """开始游戏，发送开场白"""
        if not self.players:
            return "请先分配角色再开始游戏"

        intro = talk_with_llm(f"请生成《{self.script_name}》的游戏开场白，介绍故事背景和当前场景")
        self.current_phase = "introduction"
        return intro

    def get_role_introduction(self, player_id):
        """获取角色介绍和初始信息"""
        if player_id not in self.players:
            return "玩家未找到"

        role = self.players[player_id]["role"]
        intro = talk_with_llm(f"请详细介绍《{self.script_name}》中'{role}'角色的背景、动机和初始信息")

        # 更新玩家知识
        self.players[player_id]["knowledge"].append({
            "type": "role_info",
            "content": intro
        })

        return intro

    def process_player_action(self, player_id, action_text):
        """
        处理玩家行动

        Args:
            player_id: 玩家ID
            action_text: 玩家行动描述

        Returns:
            主持人响应
        """
        if player_id not in self.players:
            return "玩家未找到"

        role = self.players[player_id]["role"]
        current_scene = self.game_state["current_scene"]

        # 构建上下文
        context = f"""
        剧本：《{self.script_name}》
        当前场景：{current_scene}
        角色：{role}
        已知线索：{', '.join(self.game_state['revealed_clues'])}
        玩家行动：{action_text}

        请根据剧本内容，以游戏主持人的身份回应这个行动。如果玩家发现了新线索，请在回应中明确指出。
        """

        response = talk_with_llm(context)

        # 分析回应中是否包含新线索
        new_clues = self._extract_clues(response)
        for clue in new_clues:
            if clue not in self.game_state["revealed_clues"]:
                self.game_state["revealed_clues"].append(clue)

        # 记录交互
        self.game_state["completed_interactions"].append({
            "player": role,
            "action": action_text,
            "response": response
        })

        # 更新游戏进度
        self._update_game_progress()

        return response

    def _extract_clues(self, text):
        """从文本中提取线索"""
        # 使用LLM识别文本中的线索
        clues_text = talk_with_llm(f"请从以下文本中提取所有重要线索，以逗号分隔列出:\n{text}")
        return [clue.strip() for clue in clues_text.split(',') if clue.strip()]

    def _update_game_progress(self):
        """更新游戏进度"""
        # 基于已揭示的线索数量和关键场景互动来计算进度
        total_clues = len(talk_with_llm(f"请列出《{self.script_name}》中所有关键线索，以逗号分隔").split(','))
        revealed = len(self.game_state["revealed_clues"])

        if total_clues > 0:
            self.game_state["game_progress"] = min(95, int(revealed / total_clues * 100))

        # 检查是否达到结局条件
        if self.game_state["game_progress"] >= 95:
            self.current_phase = "ending"

    def trigger_event(self, event_name):
        """触发特定剧情事件"""
        event_description = talk_with_llm(f"请描述《{self.script_name}》中的'{event_name}'事件发生的场景和后果")

        # 更新游戏状态
        self.game_state["current_scene"] = event_name

        # 通知所有玩家
        notifications = {}
        for player_id, player_info in self.players.items():
            role = player_info["role"]
            notification = talk_with_llm(
                f"《{self.script_name}》中的'{event_name}'事件发生了，请描述'{role}'角色会如何感知这个事件以及得到什么信息")
            notifications[player_id] = notification

        return {
            "global_description": event_description,
            "player_notifications": notifications
        }

    def provide_hint(self, player_id):
        """为特定玩家提供线索提示"""
        if player_id not in self.players:
            return "玩家未找到"

        role = self.players[player_id]["role"]
        revealed_clues = self.game_state["revealed_clues"]

        hint = talk_with_llm(f"""
        《{self.script_name}》中，角色'{role}'目前处于'{self.game_state['current_scene']}'场景。
        已发现的线索有：{', '.join(revealed_clues)}
        请提供一个适合该角色当前情况的提示，帮助推进剧情但不直接揭示谜底。
        """)

        return hint

    def end_game(self):
        """结束游戏，生成结局"""
        ending = talk_with_llm(f"""
        《{self.script_name}》游戏即将结束。
        已发现的线索：{', '.join(self.game_state['revealed_clues'])}
        请生成一个完整的结局解析，揭示真相并解释所有线索的关联。
        """)

        self.current_phase = "completed"
        return ending

    def get_character_relationships(self):
        """获取角色关系图"""
        relationships = talk_with_llm(f"请详细描述《{self.script_name}》中所有角色之间的关系")
        return relationships

    def get_game_status(self):
        """获取当前游戏状态概览"""
        return {
            "current_phase": self.current_phase,
            "scene": self.game_state["current_scene"],
            "progress": self.game_state["game_progress"],
            "revealed_clues": self.game_state["revealed_clues"],
            "player_count": len(self.players)
        }

if __name__ == "__main__":
    sh = DMDemo("年轮")
    sh.start_game()