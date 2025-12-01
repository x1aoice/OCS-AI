import json
import logging
import os
import re
import socket
import sqlite3
import threading
from typing import List, Optional, Tuple
import yaml
from fastapi import Body, FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("ocs-llm")
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CONCURRENT_WAIT_TIMEOUT = 30
PATTERN_ANSWER_PREFIX = re.compile(r'^(答案[是为：]?|答[：]?)\s*', re.IGNORECASE)
PATTERN_OPTION_PREFIX = re.compile(r'^[选项答案ABCDEFGHIJKLMNOPQRSTUVWXYZ][：:、.]\s*', re.IGNORECASE)
PATTERN_NUMBERING = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩]|\(\d+\)|（\d+）')
PATTERN_JSON_ANSWER = [
    re.compile(r'"answer"\s*:\s*"([^"]+)"'),
    re.compile(r'"result"\s*:\s*"([^"]+)"'),
    re.compile(r'"text"\s*:\s*"([^"]+)"'),
    re.compile(r'"content"\s*:\s*"([^"]+)"'),
    re.compile(r'"answer"\s*:\s*\["([^"]+)"\]'),
]

DEFAULT_CONFIG = {
    "port": 8060,
    "is_lan": False,
    "model": "gpt-3.5-turbo",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "timeout": 30,
    "max_tokens": 512,
    "min_answer_len": 4,
    "db_path": "cache.db",
    "internet_search": False,
    "db_cache": True,
    "system_prompt": '只允许输出 JSON 结果 {"answer":"..."}，禁止 Markdown 或额外内容。单选/多选题的选项内容必须直接输出，不输出选项字母，多个答案使用 # 连接。判断题只能输出"正确"或"错误"文字；若为英文判断题，输出 "True"/"False"。'
}

class AppConfig(BaseModel):
    port: int = Field(8060, ge=1, le=65535)
    is_lan: bool = Field(False)
    model: str
    base_url: str
    api_key: str
    timeout: int = 30
    max_tokens: int = 512
    min_answer_len: int = 4
    db_path: str = "cache.db"
    internet_search: bool = False
    db_cache: bool = True
    system_prompt: str = DEFAULT_CONFIG["system_prompt"]

def load_config() -> AppConfig:
    path = os.environ.get("CONFIG_FILE", "config.yaml")
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, sort_keys=False)
        except Exception:
            pass
        return AppConfig(**DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(raw)
        return AppConfig(**final_config)
    except Exception:
        return AppConfig(**DEFAULT_CONFIG)

cfg = load_config()
client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)

class Query(BaseModel):
    question: str = Field(...)
    options: Optional[List[str]] = Field(None)
    type: Optional[str] = Field(None)

class Answer(BaseModel):
    question: str
    answers: List[str]
    raw: Optional[dict] = None
    code: Optional[int] = Field(1)
    msg: Optional[str] = Field("答题成功")

class AnswerResponse(BaseModel):
    success: bool = Field(True)
    data: Answer = Field(...)

class CacheManager:
    def __init__(self, db_path: str = "cache.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    question TEXT NOT NULL,
                    options TEXT NOT NULL,
                    type TEXT NOT NULL,
                    answers TEXT NOT NULL,
                    PRIMARY KEY(question, options, type)
                )
                """
            )
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_question ON cache(question)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON cache(type)")
            except Exception:
                pass
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self.conn.execute("PRAGMA cache_size=-64000")
                self.conn.execute("PRAGMA temp_store=MEMORY")
            except Exception:
                pass
            self.conn.commit()

    def get_both(self, question: str, options: str, qtype: str) -> Tuple[Optional[str], Optional[str]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT answers, options FROM cache 
                WHERE question=? AND type=? AND (options=? OR options='')
                ORDER BY CASE WHEN options=? THEN 0 ELSE 1 END
                LIMIT 2
                """,
                (question, qtype, options, options),
            )
            rows = cur.fetchall()
        
        strict_result = None
        loose_result = None
        for row in rows:
            answers, opts = row
            if opts == options:
                strict_result = answers
            elif opts == "":
                loose_result = answers
        return (strict_result, loose_result)

    def save(self, question: str, options: str, qtype: str, answers: list[str]):
        answers_json = json.dumps(answers, ensure_ascii=False)
        with self.lock:
            try:
                self.conn.execute(
                    "REPLACE INTO cache(question, options, type, answers) VALUES (?,?,?,?)",
                    (question, options, qtype, answers_json),
                )
                self.conn.commit()
            except Exception:
                pass

_db_path = os.environ.get("CACHE_DB") or cfg.db_path
cache = CacheManager(_db_path) if cfg.db_cache else None
_pending_requests: dict[str, threading.Event] = {}
_request_results: dict[str, Answer] = {}
_global_lock = threading.Lock()

def option_label(idx: int) -> str:
    label = ""
    while True:
        idx, rem = divmod(idx, 26)
        label = chr(ord("A") + rem) + label
        if idx == 0:
            break
        idx -= 1
    return label

def auto_detect_question_type(question: str, options: Optional[List[str]] = None) -> str:
    if not question:
        return ""
    question_lower = question.lower()
    question_stripped = question.strip()
    judgement_keywords = ["是否正确", "是否", "对错", "判断", "正确", "错误", "true", "false", "√", "×", "✓", "✗"]
    if any(kw in question for kw in ["正确", "错误", "对错", "是非"]):
        return "judgement"
    if any(kw in question_lower for kw in judgement_keywords):
        return "judgement"
    if options:
        multiple_keywords = ["全选", "以上都是", "以上都对", "all", "以上"]
        options_text = " ".join(options).lower()
        if any(kw in options_text for kw in multiple_keywords):
            return "multiple"
        if any(kw in question_lower for kw in ["多选", "哪些", "哪些是", "哪些属于", "可以", "可能"]):
            return "multiple"
        return "single"
    completion_keywords = ["填空", "填入", "填写", "补充", "____", "______", "（", "）", "（  ）"]
    if any(kw in question_stripped for kw in completion_keywords):
        return "completion"
    line_keywords = ["连线", "连接", "匹配", "对应"]
    if any(kw in question_lower for kw in line_keywords):
        return "line"
    reader_keywords = ["阅读", "根据", "材料", "文章", "段落", "文本"]
    if any(kw in question_lower for kw in reader_keywords):
        return "reader"
    fill_keywords = ["完形", "完型"]
    if any(kw in question_lower for kw in fill_keywords):
        return "fill"
    return ""

def needs_detailed_answer(question: str) -> bool:
    if not question:
        return False
    detail_keywords = [
        "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
        "计算", "求", "多少", "需要", "步骤", "过程", "如何", "怎样",
        "公式", "推导", "证明", "说明", "解释", "分析"
    ]
    if any(kw in question for kw in detail_keywords):
        return True
    return PATTERN_NUMBERING.search(question) is not None

def build_prompt(q: Query) -> str:
    qtype = (q.type or "").lower()
    needs_detail = needs_detailed_answer(q.question or "")
    parts = []
    parts.append("【重要】输出格式要求：")
    parts.append("1. 必须输出纯 JSON 格式：{\"answer\":\"答案内容\"}")
    parts.append("2. 禁止使用 Markdown 代码块（如 ```json）")
    parts.append("3. 禁止转义字符，直接输出 JSON")
    parts.append("4. 不要重复题目内容，只输出答案")
    parts.append("")
    type_prompts = {
        "single": "【单选题规则】必须从给定选项中选择一个，返回对应选项的完整内容（不是选项字母 A/B/C）。",
        "multiple": "【多选题规则】必须从给定选项中选择一个或多个，返回对应选项的完整内容（不是选项字母），多个答案用 # 连接。",
        "judgement": "【判断题规则】只能返回\"正确\"或\"错误\"。英语判断题返回 \"True\" 或 \"False\"。不要返回其他内容。",
        "completion": "【填空题规则】直接返回填空内容，多个空使用 ### 连接。只返回答案，不要包含题目。",
        "line": "【连线题规则】直接返回连线结果，多个结果使用 ### 连接。",
        "fill": "【完形填空题规则】直接返回填空内容，多个空使用 ### 连接。",
        "reader": "【阅读理解题规则】直接返回答案，多个答案使用 ### 连接。",
    }
    type_prompt = type_prompts.get(qtype, "")
    if type_prompt:
        parts.append(type_prompt)
        parts.append("")
    if needs_detail:
        parts.append("【详细解答要求】")
        parts.append("题目包含多个小问或需要计算步骤，请提供详细解答：")
        parts.append("- 对于每个小问，必须保留序号（如①、②、③或(1)、(2)、(3)）")
        parts.append("- 格式：序号 简要计算过程或说明 ### 最终答案")
        parts.append("- 多个小问的答案用 ### 分隔")
        parts.append("")
    if any(k in (q.question or "") for k in ["原理", "场景", "原因", "适用"]):
        parts.append("【原理/场景题要求】")
        parts.append("若题目要求原理/原因/场景/步骤，请用简短中文描述要点；涉及原理/适用场景时，两者都要一句说明；有多个要点用 # 连接。")
        parts.append("")
    if cfg.internet_search:
        parts.append("【联网搜索】已启用模型联网搜索，可先检索再作答。")
        parts.append("")
    parts.append("【题目内容】")
    parts.append(f"题目：{q.question.strip()}")
    need_options = qtype in {"single", "multiple", "judgement"}
    if need_options and q.options:
        opts = "\n".join(f"{option_label(i)}. {opt}" for i, opt in enumerate(q.options))
        parts.append("选项：")
        parts.append(opts)
    if qtype:
        parts.append(f"题型：{qtype}")
    if not needs_detail:
        parts.append("")
        parts.append("请直接给出答案，不要解释过程。")
    parts.append("")
    parts.append("【输出示例】")
    if qtype == "judgement":
        parts.append('正确输出：{"answer":"正确"}')
    elif qtype == "single" and q.options:
        parts.append(f'正确输出：{{"answer":"{q.options[0]}"}}')
    elif qtype == "multiple":
        parts.append('正确输出：{"answer":"选项1#选项2"}')
    elif qtype in {"completion", "line", "fill", "reader"}:
        parts.append('正确输出：{"answer":"答案1###答案2"}')
    else:
        parts.append('正确输出：{"answer":"答案内容"}')
    return "\n".join(parts)

def call_llm(prompt: str) -> str:
    try:
        kwargs = {"max_tokens": cfg.max_tokens}
        if cfg.internet_search:
            kwargs["extra_body"] = {"internet_search": True}
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise HTTPException(status_code=502, detail="Empty response")
    return content

def normalize_judgement(reply: str) -> Optional[str]:
    text = reply.strip()
    text_lower = text.lower()
    yes_tokens = {
        "t", "true", "yes", "y", "1", "correct", "right", 
        "正确", "对", "是", "对的", "真", "v"
    }
    no_tokens = {
        "f", "false", "no", "n", "0", "wrong", "incorrect", 
        "错误", "错", "否", "不对", "假", "x"
    }
    if "√" in text or "✓" in text:
        return "正确"
    if "×" in text or "✗" in text:
        return "错误"
    tokens = re.split(r"[^\w\u4e00-\u9fa5]+", text_lower)
    if tokens:
        if tokens[0] in yes_tokens:
            return "正确"
        if tokens[0] in no_tokens:
            return "错误"
    if "正确" in text or "对" in text:
        return "正确"
    if "错" in text or "误" in text:
        return "错误"
    if any(kw in text_lower for kw in ["true", "yes", "correct", "right"]):
        return "正确"
    if any(kw in text_lower for kw in ["false", "no", "wrong", "incorrect"]):
        return "错误"
    return None

def extract_option_labels(reply: str, valid_labels: List[str]) -> List[str]:
    if not valid_labels:
        return []
    tokens = re.findall(r"[A-Za-z]+", reply.upper())
    allowed = {lbl.upper() for lbl in valid_labels}
    max_len = max((len(lbl) for lbl in allowed), default=0)
    seen = set()
    result = []
    for token in tokens:
        if token in allowed and token not in seen:
            seen.add(token)
            result.append(token)
            continue
        if max_len == 1:
            for ch in token:
                if ch in allowed and ch not in seen:
                    seen.add(ch)
                    result.append(ch)
    return result

def extract_option_content(reply: str, options: List[str]) -> List[str]:
    reply_lower = reply.lower()
    matched = []
    seen = set()
    for opt in options:
        opt_clean = opt.strip()
        if not opt_clean or opt_clean in seen:
            continue
        opt_lower = opt_clean.lower()
        if len(opt_clean) >= 2:
            if opt_lower in reply_lower or reply_lower in opt_lower:
                matched.append(opt_clean)
                seen.add(opt_clean)
        elif opt_clean in reply:
            matched.append(opt_clean)
            seen.add(opt_clean)
    return matched

def split_completion(reply: str) -> List[str]:
    if "###" in reply:
        parts = reply.split("###")
    else:
        parts = re.split(r"[\n;；,，、#]+", reply)
    result = [p.strip() for p in parts if p.strip()]
    if not result:
        parts = re.split(r'[，。；：、\s]+', reply)
        result = [p.strip() for p in parts if p.strip()]
    return result

def parse_cached_answer(cached_data: str) -> List[str]:
    if cached_data is None:
        return []
    try:
        parsed = json.loads(cached_data)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, dict):
            return [str(v) for v in parsed.values()]
        return [str(parsed)]
    except Exception:
        return [str(cached_data)]

def parse_llm_answer(answer_text: str, payload: Query) -> List[str]:
    qtype_lower = (payload.type or "").lower()
    if qtype_lower == "judgement":
        norm = normalize_judgement(answer_text)
        if norm:
            return [norm]
    parsed_obj = None
    try:
        parsed_obj = json.loads(answer_text)
    except json.JSONDecodeError:
        parsed_obj = None
    if parsed_obj is not None:
        answers: List[str] = []
        if isinstance(parsed_obj, dict):
            for key in ["answer", "result", "text", "content"]:
                if key in parsed_obj:
                    value = parsed_obj[key]
                    if isinstance(value, list):
                        answers = [str(x) for x in value if x]
                    else:
                        answers = [str(value)]
                    break
            if not answers and len(parsed_obj) == 1:
                answers = [str(list(parsed_obj.values())[0])]
        elif isinstance(parsed_obj, list):
            answers = [str(x) for x in parsed_obj if x]
        elif isinstance(parsed_obj, str):
            answers = [parsed_obj]
        else:
            answers = [str(parsed_obj)]
        if qtype_lower == "judgement":
            for ans in answers:
                norm = normalize_judgement(ans)
                if norm:
                    return [norm]
        if answers:
            return answers
    fallback_ans = extract_json_answer_fallback(answer_text)
    if fallback_ans:
        if qtype_lower == "judgement":
            norm = normalize_judgement(fallback_ans)
            if norm:
                return [norm]
        return [fallback_ans]
    return parse_model_reply(payload, answer_text)

def extract_json_answer_fallback(reply: str) -> Optional[str]:
    for pattern in PATTERN_JSON_ANSWER:
        m = pattern.search(reply)
        if m:
            return m.group(1)
    return None

def extract_final_answer(detailed_answer: str, question: str = "") -> str:
    # 仅保留答案自身带的序号，不再根据题干自动补序号，避免误加 ①
    if "###" not in detailed_answer:
        m = PATTERN_NUMBERING.match(detailed_answer.strip())
        if m:
            return m.group(0) + " " + detailed_answer.strip()[m.end():].strip()
        return detailed_answer.strip()

    parts = detailed_answer.split("###")
    final_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        number_prefix = ""
        number_match = PATTERN_NUMBERING.match(part)
        if number_match:
            number_prefix = number_match.group(0) + " "
            part = part[number_match.end():].strip()

        if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', part):
            final_parts.append(number_prefix + part)
        elif len(part) < 50:
            final_parts.append(number_prefix + part)
        else:
            sentences = re.split(r'[。！？\n]', part)
            found_answer = False
            for sent in reversed(sentences):
                sent = sent.strip()
                if sent and (re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', sent) or len(sent) < 30):
                    final_parts.append(number_prefix + sent)
                    found_answer = True
                    break
            if not found_answer and sentences:
                last_sentence = sentences[-1].strip()
                if last_sentence:
                    final_parts.append(number_prefix + last_sentence)
                elif part:
                    final_parts.append(number_prefix + part[:50])
    return "###".join(final_parts) if final_parts else detailed_answer.strip()

def clean_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = PATTERN_ANSWER_PREFIX.sub('', answer)
    answer = PATTERN_OPTION_PREFIX.sub('', answer)
    return answer.strip('。，、；：').strip()

def parse_model_reply(q: Query, reply: str) -> List[str]:
    qtype = (q.type or "").lower()
    if qtype == "judgement":
        norm = normalize_judgement(reply)
        if norm:
            return [norm]
        if "正确" in reply or "对" in reply or "true" in reply.lower() or "yes" in reply.lower():
            return ["正确"]
        if "错误" in reply or "错" in reply or "false" in reply.lower() or "no" in reply.lower():
            return ["错误"]
    if qtype in {"completion", "line", "fill", "reader"}:
        if needs_detailed_answer(q.question or ""):
            cleaned = extract_final_answer(reply, q.question or "")
            if cleaned:
                return [cleaned]
        parts = split_completion(reply)
        if parts:
            cleaned_parts = [clean_answer(p) for p in parts if clean_answer(p)]
            if cleaned_parts:
                return ["###".join(cleaned_parts)]
        cleaned = clean_answer(reply)
        if cleaned:
            return [cleaned]
    if qtype in {"single", "multiple"} and q.options:
        labels = [option_label(i) for i in range(len(q.options))]
        picked_labels = extract_option_labels(reply, labels)
        if picked_labels:
            contents = []
            for lbl in picked_labels:
                idx = labels.index(lbl)
                if idx < len(q.options):
                    contents.append(q.options[idx])
            if contents:
                return ["#".join(contents)] if qtype == "multiple" else [contents[0]]
        matched_contents = extract_option_content(reply, q.options)
        if matched_contents:
            return [matched_contents[0]] if qtype == "single" else ["#".join(matched_contents)]
        cleaned = clean_answer(reply)
        if cleaned and cleaned in q.options:
            return [cleaned]
        return [cleaned] if cleaned else [reply.strip()]
    cleaned = clean_answer(reply)
    return [cleaned] if cleaned else [reply]

app = FastAPI()

def _search_impl(payload: Query) -> Answer:
    if not payload.type:
        detected_type = auto_detect_question_type(payload.question, payload.options)
        if detected_type:
            payload.type = detected_type
    options_key = json.dumps(payload.options or [], ensure_ascii=False)
    qtype_key = payload.type or ""
    request_key = f"{payload.question}|{options_key}|{qtype_key}"
    with _global_lock:
        if request_key in _request_results:
            return _request_results.pop(request_key)
    if cache:
        strict_cached, loose_cached = cache.get_both(payload.question, options_key, qtype_key)
        if strict_cached:
            answers = parse_cached_answer(strict_cached)
            return Answer(
                question=payload.question,
                answers=answers,
                raw={"cache": "hit_strict"},
                code=1,
                msg="答题成功"
            )
        if loose_cached:
            answers = parse_cached_answer(loose_cached)
            return Answer(
                question=payload.question,
                answers=answers,
                raw={"cache": "hit_loose"},
                code=1,
                msg="答题成功"
            )
    is_waiter = False
    event = None
    with _global_lock:
        if request_key in _pending_requests:
            is_waiter = True
            event = _pending_requests[request_key]
        else:
            event = threading.Event()
            _pending_requests[request_key] = event
    if is_waiter:
        event.wait(timeout=CONCURRENT_WAIT_TIMEOUT)
        with _global_lock:
            if request_key in _request_results:
                result = _request_results.pop(request_key)
                _pending_requests.pop(request_key, None)
                return result
            _pending_requests.pop(request_key, None)
            event = threading.Event()
            _pending_requests[request_key] = event
            is_waiter = False
    try:
        prompt = build_prompt(payload)
        answer_text = call_llm(prompt)
        temp_answers = parse_llm_answer(answer_text, payload)
        if not temp_answers or not any(ans.strip() for ans in temp_answers):
            raise HTTPException(status_code=502, detail="LLM returned empty")
        joined_answer = "#".join(temp_answers) if temp_answers else ""
        need_retry = (len(joined_answer) < cfg.min_answer_len) and any(k in (payload.question or "") for k in ["原理","场景","原因","适用"])
        if need_retry:
            retry_prompt = prompt + "\n请补充更完整的原理和适用场景，各用一句，仍保持 JSON 输出。"
            try:
                answer_text = call_llm(retry_prompt)
                answers = parse_llm_answer(answer_text, payload)
                if not answers or not any(ans.strip() for ans in answers):
                    answers = temp_answers
            except Exception:
                answers = temp_answers
        else:
            answers = temp_answers
        if cache:
            cache.save(payload.question, options_key, qtype_key, answers)
        result = Answer(
            question=payload.question,
            answers=answers,
            raw={"model_reply": answer_text},
            code=1,
            msg="答题成功"
        )
        with _global_lock:
            _request_results[request_key] = result
            if len(_request_results) > 256:
                oldest_key = next(iter(_request_results))
                if oldest_key != request_key:
                    _request_results.pop(oldest_key, None)
            event.set()
        return result
    except Exception as e:
        error_msg = str(e)
        result = Answer(
            question=payload.question,
            answers=[],
            raw={"error": error_msg},
            code=0,
            msg=f"答题失败: {error_msg}"
        )
        with _global_lock:
            _request_results[request_key] = result
            event.set()
        return result
    finally:
        if not is_waiter:
            with _global_lock:
                _pending_requests.pop(request_key, None)

@app.post("/search", response_model=AnswerResponse)
def search_post(payload: Query = Body(...)):
    try:
        answer = _search_impl(payload)
        return AnswerResponse(success=answer.code == 1, data=answer)
    except Exception as e:
        return AnswerResponse(success=False, data=Answer(
            question=payload.question, answers=[], raw={"error": str(e)}, code=0, msg=str(e)
        ))

@app.get("/search", response_model=AnswerResponse)
def search_get(question: str, type: Optional[str] = None, options: Optional[str] = None):
    options_list = []
    if options:
        try:
            options_list = json.loads(options)
        except Exception:
            options_list = [opt.strip() for opt in options.split("\n") if opt.strip()]
    payload = Query(question=question, type=type, options=options_list if options_list else None)
    try:
        answer = _search_impl(payload)
        return AnswerResponse(success=answer.code == 1, data=answer)
    except Exception as e:
        return AnswerResponse(success=False, data=Answer(
            question=question, answers=[], raw={"error": str(e)}, code=0, msg=str(e)
        ))

def get_local_ip() -> str:
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            pass
        return socket.gethostbyname(hostname)
    except Exception:
        return '127.0.0.1'

def build_ocs_config(host: str) -> dict:
    base = f"http://{host}:{cfg.port}/search"
    return {
        "name": "OCS-AI题库",
        "url": base,
        "method": "post",
        "type": "GM_xmlhttpRequest",
        "contentType": "json",
        "headers": {"Content-Type": "application/json"},
        "data": {
            "question": "${title}",
            "options": {
                "handler": "return (env)=> env.options ? env.options.split('\\\\n') : []"
            },
            "type": {"handler": "return (env)=> env.type || ''"},
        },
        "handler": ("return (res)=>{try{if(!res || !res.data) return undefined; if(res.success && res.data.answers && Array.isArray(res.data.answers) && res.data.answers.length > 0){const arr = res.data.answers; const ans = arr.length === 1 ? arr[0] : arr.join('#'); return [res.data.question || '', ans];}if(res.data && res.data.msg) return [res.data.msg, undefined]; return undefined;}catch(e){return undefined;}}")
    }

def print_startup_info(host: str):
    config_obj = [build_ocs_config(host)]
    config_json = json.dumps(config_obj, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"OCS-AI 题库服务已启动")
    print(f"服务地址: http://{host}:{cfg.port}")
    print(f"AI 模型: {cfg.model}")
    print(f"\nOCS 配置：")
    print(config_json)
    print(f"{'='*60}\n")

def main():
    host = "0.0.0.0" if cfg.is_lan else "127.0.0.1"
    display_host = get_local_ip() if cfg.is_lan else "127.0.0.1"
    print_startup_info(display_host)
    uvicorn.run(app, host=host, port=cfg.port, log_level="warning")

if __name__ == "__main__":
    main()