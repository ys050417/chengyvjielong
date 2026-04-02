# 成语接龙游戏（Qwen3-8B GPU 版）

## 一、简介

基于 LangChain + Ollama + Qwen3 大模型实现的智能成语接龙游戏，支持GPU 加速推理、本地成语库检索、AI 校验成语合法性，提供三种游戏模式，运行流畅、体验极佳。

## 二、核心特点

| GPU 加速推理 | 支持 CUDA 显卡加速 Qwen3 模型，校验速度大幅提升 |
| :----------: | :---------------------------------------------: |
| 双重校验机制 | 本地成语库快速检索 + 大模型智能校验，准确率更高 |
| 三种游戏模式 |       玩家先手 / AI 先手 / AI 全自动对战        |
| 完整游戏逻辑 |     成语合法性校验、接龙规则判定、胜负判断      |
|   本地运行   |       无需联网，Ollama 托管模型，隐私安全       |
|  轻量化部署  |               依赖简单，配置便捷                |

## 三、环境要求

### （一）基础环境

- Python 3.8+

- Windows/Linux/macOS

### （二）拉取Qwen3模型

```
ollama pull qwen3:8b
```

## 四、部署

### （一）安装python依赖

```
pip install langchain-core langchain-ollama torch transformers
```

### （二）下载成语数据集

- 下载链接

```
https://gitcode.com/Open-source-documentation-tutorial/13d49
```

### （三）idiom_game.py

```
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import random
import time
import os

# ===================== GPU配置 =====================
# 设置Ollama使用GPU（通过环境变量）
os.environ['OLLAMA_NUM_GPU'] = '1' 
os.environ['OLLAMA_GPU_OVERHEAD'] = '1' 

# 或者使用CUDA_VISIBLE_DEVICES指定GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

# ===================== 模型：qwen3:8b（GPU加速）=====================
MODEL_NAME = "qwen3:8b"

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0.1,
    # 可选参数，根据实际情况调整
    num_gpu=1,  # 使用GPU数量，如果Ollama支持此参数
    num_thread=8,  # CPU线程数，配合GPU使用
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    USE_TRANSFORMERS = False  # 设置为True使用transformers直接加载（需要模型文件）
    if USE_TRANSFORMERS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用transformers加载模型，设备：{device}")

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # 使用合适的模型路径
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        # 如果使用transformers，需要自定义LLM包装器
        # 这里为了简化，继续使用OllamaLLM
except ImportError:
    print("⚠️ transformers未安装，继续使用OllamaLLM")


# ===================== 加载本地成语库 =====================
def load_idiom_lib(file_path="成语大全.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"❌ 未找到成语文件！错误信息：{e}")
        exit()

    idioms = re.findall(r"[\u4e00-\u9fa5]{4}", text)
    idiom_set = list(set([i.strip() for i in idioms if len(i.strip()) == 4]))
    return idiom_set, {idiom: True for idiom in idiom_set}


# 尝试加载成语文件，如果失败则使用内置示例
try:
    IDIOM_LIST, IDIOM_DICT = load_idiom_lib()
    print(f"✅ 成功加载 {len(IDIOM_LIST)} 个成语")
except Exception as e:
    print(f"❌ 加载成语文件失败：{e}")
    # 使用内置示例成语
    IDIOM_LIST = ["一心一意", "意气风发", "发奋图强", "强人所难", "难能可贵",
                  "贵耳贱目", "目中无人", "人山人海", "海阔天空", "空穴来风"]
    IDIOM_DICT = {idiom: True for idiom in IDIOM_LIST}
    print(f"✅ 使用内置示例成语库，共 {len(IDIOM_LIST)} 个成语")

# ===================== 检查GPU可用性 =====================
try:
    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU数量：{torch.cuda.device_count()}")
        print(f"✅ 当前GPU：{torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA不可用，模型将在CPU上运行")
        print("💡 如需GPU加速，请确保：")
        print("   1. 已安装CUDA和cuDNN")
        print(
            "   2. 已安装GPU版本的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Ollama已配置GPU支持")
except ImportError:
    print("⚠️ PyTorch未安装，无法检测GPU状态")
    print("💡 安装PyTorch以获得GPU支持：pip install torch")


# ===================== 使用qwen进行成语校验（GPU加速）=====================
@tool
def validate_idiom_with_qwen(idiom: str) -> dict:
    """
    使用qwen模型校验输入的内容是否为合法的四字成语
    参数:
        idiom: 需要校验的字符串
    返回:
        包含校验结果、原因、最后一个字的字典
    """
    idiom = idiom.strip()

    if len(idiom) != 4:
        return {"valid": False, "reason": "必须四字", "last": ""}

    # 先检查本地库
    if idiom in IDIOM_DICT:
        return {"valid": True, "reason": "合法", "last": idiom[-1]}

    # 使用qwen模型进行二次校验（GPU加速）
    try:
        prompt = PromptTemplate(
            template="请判断\"{idiom}\"是否为标准的四字成语。\n如果是成语，回复：是\n如果不是成语，回复：否\n只回复\"是\"或\"否\"，不要有其他内容。",
            input_variables=["idiom"]
        )

        # 使用GPU加速的LLM调用
        chain = prompt | llm | StrOutputParser()

        # 添加计时，观察GPU加速效果
        start_time = time.time()
        result = chain.invoke({"idiom": idiom}).strip()
        elapsed_time = time.time() - start_time

        if elapsed_time < 0.5:  # 只显示耗时较长的调用
            print(f"⚡ GPU加速：校验耗时 {elapsed_time:.2f} 秒")

        if result == "是":
            return {"valid": True, "reason": "合法", "last": idiom[-1]}
        else:
            return {"valid": False, "reason": "不是标准成语", "last": ""}
    except Exception as e:
        print(f"⚠️ qwen模型校验出错：{e}")
        return {"valid": False, "reason": "校验失败", "last": ""}


# ===================== 工具2：本地成语检索 =====================
@tool
def find_next_idiom_local(start_char: str) -> dict:
    """
    根据开头汉字从本地成语库中查找下一个成语
    参数:
        start_char: 成语开头的汉字
    返回:
        包含是否找到、对应成语的字典
    """
    candidates = [idiom for idiom in IDIOM_LIST if idiom[0] == start_char]
    if not candidates:
        return {"ok": False, "idiom": ""}
    chosen = random.choice(candidates)
    IDIOM_LIST.remove(chosen)
    return {"ok": True, "idiom": chosen}


# ===================== 调用链 =====================
def validate_chain(idiom):
    """简化的校验函数"""
    return validate_idiom_with_qwen.invoke(idiom)


# ===================== 模式选择 =====================
def choose_mode():
    print("\n📌 模式选择：")
    print("1. 玩家先出（人VS AI）")
    print("2. AI 先出（人VS AI）")
    print("3. AI 全自动对战（推荐）")
    while True:
        c = input("请输入 1/2/3：").strip()
        if c in ["1", "2", "3"]:
            return c


# ===================== AI VS AI =====================
def ai_vs_ai():
    print("\n🤖 AI 全自动成语接龙启动（仅使用本地成语库）！")
    print("=" * 50)

    if not IDIOM_LIST:
        print("❌ 成语库为空，无法进行游戏！")
        return

    current = random.choice(IDIOM_LIST)
    IDIOM_LIST.remove(current)
    print(f"AI1：{current}")

    round_count = 1
    while True:
        time.sleep(0.4)
        ai2 = find_next_idiom_local.invoke(current[-1])
        if not ai2["ok"]:
            print(f"\n第{round_count}轮：AI2 接不上 → AI1 胜利！🏆")
            break
        current = ai2["idiom"]
        print(f"第{round_count}轮 - AI2：{current}")

        time.sleep(0.4)
        ai1 = find_next_idiom_local.invoke(current[-1])
        if not ai1["ok"]:
            print(f"\n第{round_count}轮：AI1 接不上 → AI2 胜利！🏆")
            break
        current = ai1["idiom"]
        print(f"第{round_count}轮 - AI1：{current}")
        round_count += 1


# ===================== 人 VS AI =====================
def play_game():
    print("🎉 成语接龙（qwen3:8b模型校验，GPU加速）")
    print("✅ 本地成语库检索 + qwen模型校验")
    print("💡 输入 退出 结束游戏\n")
    print(f"📚 当前成语库大小：{len(IDIOM_LIST)} 个成语")

    if not IDIOM_LIST:
        print("❌ 成语库为空，无法开始游戏！")
        print("💡 请确保'成语大全.txt'文件存在且包含四字成语")
        return

    mode = choose_mode()

    if mode == "3":
        ai_vs_ai()
        return

    current = ""
    if mode == "1":
        while True:
            user = input("\n你先出：").strip()
            if user == "退出":
                return
            print("🔍 正在校验成语...")
            res = validate_chain(user)
            if not res["valid"]:
                print(f"❌ {res['reason']}")
                continue
            current = user
            if current in IDIOM_LIST:
                IDIOM_LIST.remove(current)
            print(f"✅ 你：{current}")
            break

    else:
        if not IDIOM_LIST:
            print("❌ 成语库为空，无法开始游戏！")
            return
        current = random.choice(IDIOM_LIST)
        IDIOM_LIST.remove(current)
        print(f"AI：{current}")

    while True:
        need = current[-1]
        user = input(f"\n请以【{need}】开头：").strip()
        if user == "退出":
            break

        print("🔍 正在校验成语...")
        res = validate_chain(user)
        if not res["valid"]:
            print(f"❌ {res['reason']}")
            break
        if user[0] != need:
            print(f"❌ 必须以【{need}】开头！你输了")
            break
        if user in IDIOM_LIST:
            IDIOM_LIST.remove(user)

        print("🤖 AI 正在检索本地成语库...")
        ai = find_next_idiom_local.invoke(user[-1])
        if not ai["ok"]:
            print("\n🎉 AI接不上，你赢了！")
            break
        current = ai["idiom"]
        print(f"AI：{current}")


# ===================== 统计信息 =====================
def show_stats():
    print(f"\n📊 剩余成语库大小：{len(IDIOM_LIST)} 个成语")
    print(f"📊 已使用的成语：{len(IDIOM_DICT) - len(IDIOM_LIST)} 个")


if __name__ == "__main__":
    try:
        # 显示GPU信息
        print("\n" + "=" * 50)
        print("🔧 系统配置信息：")
        try:
            import torch

            if torch.cuda.is_available():
                print(f"🎮 GPU加速：已启用")
                print(f"🎮 GPU型号：{torch.cuda.get_device_name(0)}")
            else:
                print(f"💻 GPU加速：未启用（使用CPU）")
        except:
            print(f"💻 GPU加速：未检测到PyTorch")
        print("=" * 50 + "\n")

        play_game()
        show_stats()
    except KeyboardInterrupt:
        print("\n\n👋 游戏被中断，感谢游玩！")
    except Exception as e:
        print(f"\n❌ 游戏出现错误：{e}")
        print("💡 请检查网络连接和qwen模型是否正常运行")
```

## 五、常见问题

### （一）模型启动失败

- 检查 Ollama 是否启动

- 执行 ollama list 查看 qwen3:8b 是否安装成功

- 重新拉取模型：ollama pull qwen3:8b

### （二）无 GPU / CUDA 不可用

- 程序自动切换为 CPU 运行，可正常使用

- 安装 GPU 版 PyTorch：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### （三）成语库加载失败

- 确保成语大全.txt与程序在同一目录