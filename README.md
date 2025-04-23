</div>

<div align="center">

![:name](https://count.getloli.com/@llmgo?name=llmgo&theme=miku&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

</div>

# LLM Go Plugin for AstrBot

这是一个围棋插件，允许用户与 LLM 进行围棋对弈。

## 功能

*   开始一局新的 19x19 围棋游戏。
*   用户可以选择执黑棋或白棋。
*   通过指令在棋盘上落子。
*   LLM (通过 AstrBot 的 LLM 提供者) 作为对手进行游戏。
*   使用 Pillow 库渲染棋盘图像。
*   支持中途退出游戏。

## 使用方法

通过 AstrBot 向机器人发送以下指令：

*   **/llmgo start &lt;颜色&gt;**: 开始一局新游戏。
    *   `<颜色>`: 必须是 `黑棋` 或 `白棋`。
    *   示例: `/llmgo start 黑棋`
*   **/llmgo place &lt;x&gt; &lt;y&gt;**: 在指定坐标落子。
    *   `<x>`: 横坐标 (0-18)。
    *   `<y>`: 纵坐标 (0-18)。
    *   示例: `/llmgo place 9 9`
*   **/llmgo quit**: 结束当前正在进行的游戏。

## 依赖项

*   **AstrBot**: 插件运行的基础框架。
*   **Pillow**: 用于图像处理和棋盘渲染。 (通常随 AstrBot 核心依赖安装)
*   **NumPy**: 用于棋盘数据表示。 (通常随 AstrBot 核心依赖安装)
*   **配置好的 LLM 提供者**: AstrBot 需要配置至少一个 LLM 提供者来驱动 AI 对手。

## 安装

1.  将 `main.py` 文件放置在 AstrBot 的 `plugins` 目录下，可以放在一个名为 `llm_go` 的子目录中（例如：`plugins/llm_go/main.py`）。
2.  确保您的 AstrBot 环境安装了必要的依赖库 (Pillow, NumPy)。如果缺少，可以通过 pip 安装：
    ```bash
    pip install Pillow numpy
    ```
3.  重启或重新加载 AstrBot 以加载插件。
4.  确保 AstrBot 配置了可用的 LLM 提供者。

## 作者

*   Jason.Joestar

## 版本

*   0.1.0
