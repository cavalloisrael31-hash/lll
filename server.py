"""数字人基础后端：ASR、Chat、TTS 及前端流式接口。

"""
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("E:/obj_render")
DEMO_WAV = Path("E:/web_demo_robot/static/common/test.wav")

app = FastAPI(title="Digital Human API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- 请求协议 --------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ProviderConfig(BaseModel):

    api_base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class ASRRequest(ProviderConfig):
    audio_base64: str = Field(min_length=1)
    audio_mime_type: str = "audio/webm"
    language: str = "zh"
    asr_model: str = "whisper-1"


class ChatRequest(ProviderConfig):
    text: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class TTSRequest(ProviderConfig):
    text: str = Field(min_length=1)
    voice: str = "alloy"
    tts_model: str = "gpt-4o-mini-tts"


class AvatarRequest(ProviderConfig):
    input_mode: Literal["text", "audio"]
    prompt: str | None = None
    audio: str | None = None
    audio_mime_type: str = "audio/webm"
    history: list[ChatMessage] = Field(default_factory=list)
    asr_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    # 由 server.py 下发；下一阶段可替换为 AI 表情/动作决策结果。
    action: Literal["idle", "waveHello", "thinking", "jump", "leftArmRaise", "aPose", "sit"] = "idle"
    action_intensity: float = Field(default=1.0, ge=0.0, le=1.0)


# -------------------- 通用配置 --------------------
def get_provider_config(body: ProviderConfig) -> tuple[str, str, str]:
    base_url = (body.api_base_url or os.getenv("AI_API_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    api_key = body.api_key or os.getenv("AI_API_KEY", "")
    model = body.model or os.getenv("AI_MODEL", "gpt-4.1-mini")
    return base_url, api_key, model


def make_api_error(response: requests.Response, service: str) -> HTTPException:
    try:
        detail = response.json().get("error", {}).get("message", response.text[:500])
    except ValueError:
        detail = response.text[:500]
    return HTTPException(502, f"{service} 调用失败：{detail}")


# -------------------- ASR 模块 --------------------
def transcribe(body: ASRRequest) -> str:
    try:
        audio_bytes = base64.b64decode(body.audio_base64, validate=True)
    except Exception as exc:
        raise HTTPException(400, f"audio_base64 无效：{exc}") from exc
    if not audio_bytes:
        raise HTTPException(400, "音频为空")

    base_url, api_key, _ = get_provider_config(body)
    if not api_key:
        return "我收到了你的语音。（当前为 ASR 演示模式）"

    response = requests.post(
        f"{base_url}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("recording.webm", audio_bytes, body.audio_mime_type)},
        data={"model": body.asr_model, "language": body.language},
        timeout=90,
    )
    if not response.ok:
        raise make_api_error(response, "ASR")
    text = str(response.json().get("text", "")).strip()
    if not text:
        raise HTTPException(502, "ASR 没有返回文字")
    return text


# -------------------- Chat 模块 --------------------
def chat(body: ChatRequest) -> str:
    """调用 OpenAI-compatible POST /chat/completions；逻辑参考 aiChat.exe。"""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "聊天内容不能为空")
    base_url, api_key, model = get_provider_config(body)
    if not api_key:
        return f"你说的是：{text}。这是数字人基础功能的演示回复。"

    messages = [message.model_dump() for message in body.history]
    messages.append({"role": "user", "content": text})
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.7},
        timeout=90,
    )
    if not response.ok:
        raise make_api_error(response, "Chat")
    try:
        answer = str(response.json()["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(502, "Chat 返回格式不正确") from exc
    if not answer:
        raise HTTPException(502, "Chat 没有返回回复")
    return answer


# -------------------- 动作/表情控制模块（第二次 LLM 调用）--------------------
ALLOWED_ACTIONS = {"idle", "waveHello", "thinking", "jump", "leftArmRaise", "aPose", "sit"}
ALLOWED_EXPRESSIONS = {"neutral", "smile", "surprise", "frown", "thinking", "blink", "pucker", "sneer"}


def default_control(reply: str) -> dict:
    """未配置模型时，仍可验证“回复文本分流到动作管理”的完整链路。"""
    if any(word in reply for word in ("高兴", "开心", "欢迎", "你好")):
        return {"action": "waveHello", "action_intensity": 0.6, "expression": "smile", "expression_intensity": 0.55}
    if any(word in reply for word in ("想", "问题", "为什么", "如何")):
        return {"action": "thinking", "action_intensity": 0.55, "expression": "thinking", "expression_intensity": 0.55}
    return {"action": "idle", "action_intensity": 1.0, "expression": "neutral", "expression_intensity": 1.0}


def decide_control(reply: str, config: ProviderConfig) -> dict:
    """第二次 Chat 调用：只把第一次 Chat 的回复转为动作/表情 JSON。"""
    base_url, api_key, model = get_provider_config(config)
    if not api_key:
        return default_control(reply)
    instruction = (
        "根据以下数字人将要说出的回复，选择一个身体动作和一个面部表情。"
        f"动作只能是 {sorted(ALLOWED_ACTIONS)}；表情只能是 {sorted(ALLOWED_EXPRESSIONS)}。"
        "只输出 JSON，不要解释："
        '{"action":"idle","action_intensity":0.0,"expression":"neutral","expression_intensity":0.0}'
    )
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": reply}], "temperature": 0.2},
        timeout=90,
    )
    if not response.ok:
        raise make_api_error(response, "动作表情控制")
    try:
        raw = str(response.json()["choices"][0]["message"]["content"])
        match = re.search(r"\{.*\}", raw, re.S)
        decision = json.loads(match.group()) if match else {}
        action = decision.get("action", "idle")
        expression = decision.get("expression", "neutral")
        if action not in ALLOWED_ACTIONS or expression not in ALLOWED_EXPRESSIONS:
            raise ValueError("控制值不在允许范围")
        clamp = lambda value: max(0.0, min(1.0, float(value)))
        return {"action": action, "action_intensity": clamp(decision.get("action_intensity", .6)), "expression": expression, "expression_intensity": clamp(decision.get("expression_intensity", .6))}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(502, f"动作表情控制返回格式错误：{exc}") from exc


# -------------------- TTS 模块 --------------------
def synthesize(body: TTSRequest) -> tuple[str, str]:
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "TTS 文本不能为空")
    base_url, api_key, _ = get_provider_config(body)
    if not api_key:
        if not DEMO_WAV.exists():
            raise HTTPException(500, f"缺少演示音频：{DEMO_WAV}")
        return base64.b64encode(DEMO_WAV.read_bytes()).decode("ascii"), "audio/wav"

    response = requests.post(
        f"{base_url}/audio/speech",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": body.tts_model, "voice": body.voice, "input": text, "response_format": "wav"},
        timeout=90,
    )
    if not response.ok:
        raise make_api_error(response, "TTS")
    return base64.b64encode(response.content).decode("ascii"), "audio/wav"


# -------------------- 给数字人前端的流式编排 --------------------
def split_sentences(text: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[。？！；…])", text) if part.strip()]
    return chunks or [text]


async def generate_avatar_events(body: AvatarRequest):
    if body.input_mode == "audio":
        if not body.audio:
            raise HTTPException(400, "audio 模式缺少 audio")
        transcript = transcribe(ASRRequest(
            audio_base64=body.audio,
            audio_mime_type=body.audio_mime_type,
            asr_model=body.asr_model,
            api_base_url=body.api_base_url,
            api_key=body.api_key,
            model=body.model,
        ))
        yield json.dumps({"type": "transcript", "prompt": transcript}, ensure_ascii=False) + "\n"
    else:
        transcript = (body.prompt or "").strip()
        if not transcript:
            raise HTTPException(400, "text 模式缺少 prompt")

    reply = chat(ChatRequest(
        text=transcript, history=body.history,
        api_base_url=body.api_base_url, api_key=body.api_key, model=body.model,
    ))
    # 第一次 LLM 输出 reply 后，分流：左侧给 TTS，右侧进行第二次 LLM 控制决策。
    tts_task = asyncio.create_task(asyncio.to_thread(synthesize, TTSRequest(text=reply, voice=body.voice, tts_model=body.tts_model, api_base_url=body.api_base_url, api_key=body.api_key, model=body.model)))
    control_task = asyncio.create_task(asyncio.to_thread(decide_control, reply, body))
    audio, audio_mime_type = await tts_task
    yield json.dumps({"type": "speech", "text": reply, "audio": audio, "audio_mime_type": audio_mime_type, "endpoint": True}, ensure_ascii=False) + "\n"
    control = await control_task
    # 输出二：前端根据该独立控制事件执行动作与表情，不参与 TTS。
    yield json.dumps({"type": "control", **control}, ensure_ascii=False) + "\n"


# -------------------- API 路由 --------------------
@app.get("/api/health")
def health():
    return {"ok": True, "modules": ["asr", "chat", "tts"], "prompt_enabled": False}


@app.post("/api/asr")
def asr_api(body: ASRRequest):
    return {"text": transcribe(body)}


@app.post("/api/chat")
def chat_api(body: ChatRequest):
    return {"text": chat(body)}


@app.post("/api/tts")
def tts_api(body: TTSRequest):
    audio, audio_mime_type = synthesize(body)
    return {"audio": audio, "audio_mime_type": audio_mime_type}


@app.post("/api/avatar/stream")
async def avatar_stream_api(body: AvatarRequest):
    return StreamingResponse(generate_avatar_events(body), media_type="application/x-ndjson; charset=utf-8")


@app.post("/eb_stream")
async def legacy_stream_api(body: AvatarRequest):
    """兼容当前 HTML 的旧接口，后续前端改用 /api/avatar/stream。"""
    return StreamingResponse(generate_avatar_events(body), media_type="application/x-ndjson; charset=utf-8")


# -------------------- 当前演示页面与 3D 资源 --------------------
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "face_expression_viewer.html")


@app.get("/renderer")
def renderer():
    return FileResponse(BASE_DIR / "face_expression_viewer.html")


@app.get("/ui.js")
def ui_module():
    return FileResponse(BASE_DIR / "ui.js", media_type="application/javascript")


@app.get("/face_expression_viewer.js")
def renderer_module():
    return FileResponse(BASE_DIR / "face_expression_viewer.js", media_type="application/javascript")


@app.get("/{asset_name}")
def model_asset(asset_name: str):
    allowed = {"obj_model.bin", "flame_arkit_bs.npy", "SMPL-X__FLAME_vertex_ids.npy"}
    if asset_name not in allowed:
        raise HTTPException(404, "Not found")
    return FileResponse(MODEL_DIR / asset_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8888")))
