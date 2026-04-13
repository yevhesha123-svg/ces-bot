import os
import re
import json
import logging
import httpx
import pdfplumber
from io import BytesIO
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters, ConversationHandler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

# ── Conversation states ────────────────────────────────────────────────────────
LONGREAD_WAIT_SOURCE = 1
LONGREAD_WAIT_LANG   = 2
LONGREAD_WAIT_TOPICS = 3
LONGREAD_WAIT_EDIT   = 4

# ── Persistent memory file ─────────────────────────────────────────────────────
MEMORY_FILE = "longread_memory.json"

def load_memory() -> list[str]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(rules: list[str]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

def add_memory_rule(rule: str):
    rules = load_memory()
    rules.append(rule)
    save_memory(rules)

# ── CES style guide extracted from 4 longreads ────────────────────────────────
CES_STYLE_GUIDE = """
STYLE GUIDE FOR CES LONGREADS — extracted from real examples on ces.org.ua:

STRUCTURE:
- Start with a 2-3 sentence hook paragraph that frames the key tension or challenge
- Podcast attribution block: "In a recent episode of the «What's Wrong with the Economy?» podcast, **Guest Name**, Title at Organisation, discussed..."
- Standard podcast line: *"What's up with the economy?" is a weekly podcast by the Centre for Economic Strategy in collaboration with Hromadske Radio and supported by PrivatBank.*
- *Hosts Anhelina Zavadetska and Maksym Samoiliuk speak with experts, entrepreneurs, analysts, and government officials about the current state of Ukraine's economy.*
- Section headers: ### **Bold numbered or thematic title**
- 2-4 paragraphs of editorial prose per section (NOT bullet points for main content)
- Direct quotes in blockquote format: > "Quote text" — always paraphrased context before the quote
- Closing paragraph that synthesises the key takeaway or call to action

TONE & VOICE:
- Authoritative but accessible — explains complex economic concepts without jargon
- Journalistic neutrality — presents expert views without editorialising
- Specific and data-driven — always mentions concrete numbers, percentages, timelines
- Urgent but measured — conveys stakes without alarmism
- Phrases like: "The consequences are already physical:", "A major point of contention lies in...", "The expert emphasises that...", "[Name] points out that...", "The core of the disagreement lies in..."

FORMATTING RULES:
- Guest name always **bold** on first mention
- Quotes introduced with attribution: "As [Name] explains:", "According to [Name]:", "[Name] emphasises:", "[Name] points out:"
- Numbers formatted: 17,000 (with comma), $60–$90 (en dash), 2022–2026
- Section count: 4-6 sections typical
- Length: 600-900 words typical for English version

WHAT TO AVOID:
- No bullet points for main content (only for clustered data like research findings)
- No "In conclusion" or "To summarise"
- No first-person ("we think", "we believe")
- No vague language — always anchor to specific facts from the transcript
"""

# ── Transcript system prompts ──────────────────────────────────────────────────
TRANSCRIPT_SYSTEM = """Ти — асистент подкасту «Що з економікою?» Центру економічної стратегії (ЦЕС).

Твоє завдання — перетворювати сирий транскрипт подкасту на відредаговану текстову розшифровку для публікації на сайті ЦЕС.

ПРАВИЛА ФОРМАТУВАННЯ:
1. Кожен спікер виділяється жирним: **Ім'я Прізвище:**
2. Типові спікери: Ангеліна Завадецька, Максим Самойлюк — і гість епізоду (його ім'я визнач з контексту)
3. Прибирай прев'ю/тизер на початку — транскрипт починається з привітання Ангеліни
4. Виправляй граматичні та стилістичні помилки автотранскрибації
5. Виправляй помилкові назви
6. Прибирай заминки, повтори, слова-паразити — але зберігай суть і стиль мовлення
7. Правильно розподіляй репліки між спікерами
8. Зберігай природний розмовний стиль
9. НЕ додавай нічого від себе
10. Починай одразу з тексту, без вступних коментарів

СТРУКТУРА ВИВОДУ:
**Ім'я Прізвище:** Текст репліки"""

CHUNK_SIZE = 15000
STREAM_UPDATE_EVERY = 300


# ── Claude API calls ───────────────────────────────────────────────────────────

async def call_claude_streaming(system: str, user_message: str, on_chunk) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 8192,
                "stream": True,
                "system": system,
                "messages": [{"role": "user", "content": user_message}]
            }
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                data = json.loads(body)
                error_msg = data.get("error", {}).get("message", str(data))
                raise ValueError(f"Anthropic API error {response.status_code}: {error_msg}")

            full_text = ""
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except Exception:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {}).get("text", "")
                    if delta:
                        full_text += delta
                        await on_chunk(delta)
            return full_text


async def call_claude_simple(system: str, user_message: str) -> str:
    """Non-streaming call for short tasks like topic extraction."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "system": system,
                "messages": [{"role": "user", "content": user_message}]
            }
        )
        data = response.json()
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", str(data))
            raise ValueError(f"Anthropic API error {response.status_code}: {error_msg}")
        return data["content"][0]["text"]


# ── File helpers ───────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        split_at = text.rfind('\n\n', 0, chunk_size)
        if split_at == -1:
            split_at = text.rfind('\n', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


async def fetch_url_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
        # Strip scripts, styles, nav, footer
        html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
        html = re.sub(r"(?is)<nav[^>]*>.*?</nav>", " ", html)
        html = re.sub(r"(?is)<footer[^>]*>.*?</footer>", " ", html)
        html = re.sub(r"(?is)<header[^>]*>.*?</header>", " ", html)
        # Try to get article content
        m = re.search(r"(?is)<article[^>]*>(.*?)</article>", html)
        if m:
            html = m.group(1)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def extract_pdf_text(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def markdown_to_html(text: str) -> str:
    lines = text.split("\n")
    html_lines = []
    for line in lines:
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"\*(.+?)\*", r"<i>\1</i>", line)
        # Blockquotes
        if line.strip().startswith("> "):
            line = f'<blockquote>{line.strip()[2:]}</blockquote>'
            html_lines.append(line)
            continue
        # Headers
        if line.strip().startswith("### "):
            line = f'<h3>{line.strip()[4:]}</h3>'
            html_lines.append(line)
            continue
        if line.strip():
            html_lines.append(f"<p>{line}</p>")
        else:
            html_lines.append("<br>")
    body = "\n".join(html_lines)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Georgia, serif; font-size: 16px; line-height: 1.8;
           max-width: 820px; margin: 40px auto; padding: 0 24px; color: #222; }}
    h3 {{ font-size: 18px; margin: 32px 0 12px; }}
    p {{ margin: 0 0 12px; }}
    blockquote {{ border-left: 3px solid #ccc; margin: 16px 0; padding: 8px 16px;
                  color: #444; font-style: italic; }}
    b {{ color: #000; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


async def send_as_file(update: Update, text: str, filename: str):
    html_content = markdown_to_html(text)
    file_bytes = html_content.encode("utf-8")
    html_filename = re.sub(r'[^\w\-.]', '_', filename) + ".html"
    await update.message.reply_document(
        document=BytesIO(file_bytes),
        filename=html_filename,
        caption="✅ Готово! Відкрийте файл у браузері та скопіюйте текст на сайт ЦЕС."
    )


# ── Transcript processing ──────────────────────────────────────────────────────

async def process_transcript(update: Update, raw_text: str, source_name: str):
    chunks = split_into_chunks(raw_text)
    total = len(chunks)
    all_results = []

    if total > 1:
        await update.message.reply_text(
            f"📋 Текст великий — розбиваю на {total} частини..."
        )

    for i, chunk in enumerate(chunks, 1):
        label = f"частина {i}/{total}" if total > 1 else "транскрипт"
        live_msg = await update.message.reply_text(f"✍️ Редагую {label}...\n\n_(текст з'явиться тут)_")

        full_result = ""
        chars_since_update = 0

        async def on_chunk(delta):
            nonlocal full_result, chars_since_update
            full_result += delta
            chars_since_update += len(delta)
            if chars_since_update >= STREAM_UPDATE_EVERY:
                preview = full_result[-1500:] if len(full_result) > 1500 else full_result
                try:
                    await live_msg.edit_text(f"✍️ Редагую {label}...\n\n{preview}")
                except Exception:
                    pass
                chars_since_update = 0

        full_result = await call_claude_streaming(TRANSCRIPT_SYSTEM, 
            f"Відредагуй транскрипт:\n\n{chunk}", on_chunk)
        all_results.append(full_result)

        try:
            preview = full_result[-1500:] if len(full_result) > 1500 else full_result
            await live_msg.edit_text(f"✅ {label.capitalize()} готова!\n\n{preview}")
        except Exception:
            pass

    full_text = "\n\n".join(all_results)
    name = source_name.replace(".pdf", "").replace(".txt", "") + "_транскрипт"
    await send_as_file(update, full_text, name)


# ── Longread conversation ──────────────────────────────────────────────────────

async def longread_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "📝 *Генератор лонгрідів ЦЕС*\n\n"
        "Надішліть мені джерело:\n"
        "• PDF або TXT файл з транскриптом\n"
        "• Посилання на сторінку з текстом\n\n"
        "Або напишіть /cancel щоб скасувати.",
        parse_mode="Markdown"
    )
    return LONGREAD_WAIT_SOURCE


async def longread_got_source(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw_text = ""
    source_name = "longread"

    # Text message — could be URL or plain text
    if update.message.text:
        text = update.message.text.strip()
        urls = re.findall(r'https?://\S+', text)
        if urls:
            await update.message.reply_text("⏳ Завантажую сторінку...")
            try:
                raw_text = await fetch_url_text(urls[0])
                source_name = re.sub(r'https?://', '', urls[0]).replace('/', '_')[:40]
            except Exception as e:
                await update.message.reply_text(f"❌ Не вдалося завантажити: {e}")
                return LONGREAD_WAIT_SOURCE
        else:
            raw_text = text
            source_name = "longread"

    # Document
    elif update.message.document:
        doc = update.message.document
        fname = doc.file_name.lower()
        if not fname.endswith(".pdf") and not fname.endswith(".txt"):
            await update.message.reply_text("⚠️ Надішліть PDF або TXT файл.")
            return LONGREAD_WAIT_SOURCE
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()
        try:
            raw_text = bytes(file_bytes).decode("utf-8", errors="ignore") if fname.endswith(".txt") \
                       else extract_pdf_text(bytes(file_bytes))
            source_name = doc.file_name.replace(".pdf", "").replace(".txt", "")
        except Exception as e:
            await update.message.reply_text(f"❌ Не вдалося прочитати файл: {e}")
            return LONGREAD_WAIT_SOURCE

    if not raw_text.strip():
        await update.message.reply_text("❌ Текст порожній. Спробуйте ще раз.")
        return LONGREAD_WAIT_SOURCE

    context.user_data['raw_text'] = raw_text
    context.user_data['source_name'] = source_name

    # Ask language
    await update.message.reply_text(
        "🌐 Оберіть мову лонгріду:\n\n"
        "🇺🇦 Напишіть *ua* — українська\n"
        "🇬🇧 Напишіть *en* — англійська",
        parse_mode="Markdown"
    )
    return LONGREAD_WAIT_LANG


async def longread_got_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = update.message.text.strip().lower()
    if lang not in ("ua", "en", "🇺🇦", "🇬🇧"):
        await update.message.reply_text("Напишіть *ua* або *en*", parse_mode="Markdown")
        return LONGREAD_WAIT_LANG

    context.user_data['lang'] = "ukrainian" if lang in ("ua", "🇺🇦") else "english"

    # Auto-extract topics from transcript
    await update.message.reply_text("⏳ Аналізую текст і визначаю ключові теми...")

    raw_text = context.user_data['raw_text']
    preview = raw_text[:8000]

    try:
        topics_raw = await call_claude_simple(
            "You are an analyst. Extract 5-7 key topics from this podcast transcript. "
            "Return ONLY a numbered list, one topic per line, no extra text.",
            f"Transcript:\n\n{preview}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")
        return ConversationHandler.END

    context.user_data['suggested_topics'] = topics_raw

    await update.message.reply_text(
        f"📋 *Запропоновані теми:*\n\n{topics_raw}\n\n"
        "Ви можете:\n"
        "• Написати *ок* — використати ці теми\n"
        "• Написати свої теми (просто перерахуйте їх)\n"
        "• Додати до існуючих: *+ ваша тема*",
        parse_mode="Markdown"
    )
    return LONGREAD_WAIT_TOPICS


async def longread_got_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    suggested = context.user_data.get('suggested_topics', '')

    if user_input.lower() in ("ок", "ok", "ок.", "ok."):
        final_topics = suggested
    elif user_input.startswith("+"):
        extra = user_input[1:].strip()
        final_topics = suggested + f"\n{extra}"
    else:
        final_topics = user_input

    context.user_data['final_topics'] = final_topics
    lang = context.user_data['lang']
    raw_text = context.user_data['raw_text']
    source_name = context.user_data['source_name']

    # Load persistent memory rules
    memory_rules = load_memory()
    memory_block = ""
    if memory_rules:
        rules_text = "\n".join(f"- {r}" for r in memory_rules)
        memory_block = f"\n\nADDITIONAL RULES FROM PREVIOUS FEEDBACK:\n{rules_text}"

    lang_instruction = "Write the longread in ENGLISH." if lang == "english" \
                       else "Напиши лонгрід УКРАЇНСЬКОЮ мовою."

    system_prompt = f"""{CES_STYLE_GUIDE}

{lang_instruction}
{memory_block}

Your task: write a longread article for the CES website based on the provided podcast transcript.
Follow the CES style guide strictly. Use ONLY information from the transcript — do not invent facts.
Structure the article around the provided topics, but feel free to add 1-2 more if the transcript warrants it.
Include direct quotes from the guest where relevant (keep them under 50 words each).
Output only the article text, starting directly with the opening paragraph."""

    user_message = f"""TOPICS TO COVER:
{final_topics}

TRANSCRIPT:
{raw_text[:20000]}"""

    await update.message.reply_text("✍️ Генерую лонгрід...")

    live_msg = await update.message.reply_text("_(текст з'явиться тут)_")
    full_result = ""
    chars_since_update = 0

    async def on_chunk(delta):
        nonlocal full_result, chars_since_update
        full_result += delta
        chars_since_update += len(delta)
        if chars_since_update >= STREAM_UPDATE_EVERY:
            preview = full_result[-2000:] if len(full_result) > 2000 else full_result
            try:
                await live_msg.edit_text(f"✍️ Пишу...\n\n{preview}")
            except Exception:
                pass
            chars_since_update = 0

    try:
        full_result = await call_claude_streaming(system_prompt, user_message, on_chunk)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {type(e).__name__}: {e}")
        return ConversationHandler.END

    try:
        await live_msg.edit_text("✅ Лонгрід готовий!")
    except Exception:
        pass

    context.user_data['longread_result'] = full_result

    # Send as file
    await send_as_file(update, full_result, source_name + "_longread")

    await update.message.reply_text(
        "💬 *Що далі?*\n\n"
        "• Напишіть правки — і я переписую лонгрід\n"
        "• *Запам'ятай: [правило]* — збережу правило для всіх майбутніх лонгрідів\n"
        "• /done — завершити\n"
        "• /longread — новий лонгрід",
        parse_mode="Markdown"
    )
    return LONGREAD_WAIT_EDIT


async def longread_got_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()

    # Save persistent rule
    if user_input.lower().startswith("запам'ятай:") or user_input.lower().startswith("запам'ятай:"):
        rule = re.sub(r"(?i)запам['']ятай:\s*", "", user_input).strip()
        add_memory_rule(rule)
        await update.message.reply_text(
            f"✅ Запам'ятав! Це правило застосовуватиметься до всіх наступних лонгрідів:\n_{rule}_",
            parse_mode="Markdown"
        )
        return LONGREAD_WAIT_EDIT

    if user_input.lower() in ("/done", "done", "готово"):
        await update.message.reply_text("✅ Завершено! Для нового лонгріду — /longread")
        return ConversationHandler.END

    # Apply edit
    current_longread = context.user_data.get('longread_result', '')
    raw_text = context.user_data.get('raw_text', '')
    lang = context.user_data.get('lang', 'english')
    lang_instruction = "Keep the article in ENGLISH." if lang == "english" \
                       else "Залиш статтю УКРАЇНСЬКОЮ мовою."

    memory_rules = load_memory()
    memory_block = ""
    if memory_rules:
        rules_text = "\n".join(f"- {r}" for r in memory_rules)
        memory_block = f"\nADDITIONAL RULES:\n{rules_text}"

    system_prompt = f"""{CES_STYLE_GUIDE}
{lang_instruction}
{memory_block}
You are editing an existing CES longread based on feedback. Apply the requested changes.
Keep everything else the same. Output only the full updated article."""

    user_message = f"""CURRENT ARTICLE:
{current_longread}

EDIT REQUEST:
{user_input}

ORIGINAL TRANSCRIPT (for reference):
{raw_text[:10000]}"""

    await update.message.reply_text("✍️ Вношу правки...")
    live_msg = await update.message.reply_text("_(оновлений текст з'явиться тут)_")

    full_result = ""
    chars_since_update = 0

    async def on_chunk(delta):
        nonlocal full_result, chars_since_update
        full_result += delta
        chars_since_update += len(delta)
        if chars_since_update >= STREAM_UPDATE_EVERY:
            preview = full_result[-2000:] if len(full_result) > 2000 else full_result
            try:
                await live_msg.edit_text(f"✍️ Редагую...\n\n{preview}")
            except Exception:
                pass
            chars_since_update = 0

    try:
        full_result = await call_claude_streaming(system_prompt, user_message, on_chunk)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {type(e).__name__}: {e}")
        return LONGREAD_WAIT_EDIT

    context.user_data['longread_result'] = full_result

    try:
        await live_msg.edit_text("✅ Правки внесено!")
    except Exception:
        pass

    source_name = context.user_data.get('source_name', 'longread')
    await send_as_file(update, full_result, source_name + "_longread_v2")

    await update.message.reply_text(
        "💬 Ще правки? Або:\n• *Запам'ятай: [правило]* — зберегти правило\n• /done — завершити",
        parse_mode="Markdown"
    )
    return LONGREAD_WAIT_EDIT


async def longread_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ Скасовано.")
    return ConversationHandler.END


# ── Transcript handlers ────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Я бот ЦЕС.\n\n"
        "📄 Надішли PDF або TXT — отримаєш відредагований транскрипт\n"
        "📝 /longread — створити лонгрід по подкасту\n"
        "🧠 /memory — переглянути збережені правила\n"
        "🗑 /clearmemory — очистити правила"
    )


async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rules = load_memory()
    if not rules:
        await update.message.reply_text("🧠 Збережених правил немає.")
        return
    text = "🧠 *Збережені правила для лонгрідів:*\n\n"
    for i, r in enumerate(rules, 1):
        text += f"{i}. {r}\n"
    await update.message.reply_text(text, parse_mode="Markdown")


async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_memory([])
    await update.message.reply_text("🗑 Всі правила видалено.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    fname = doc.file_name.lower()
    if not fname.endswith(".pdf") and not fname.endswith(".txt"):
        await update.message.reply_text("⚠️ Надішліть PDF або TXT файл.")
        return
    await update.message.reply_text("⏳ Читаю файл...")
    file = await context.bot.get_file(doc.file_id)
    file_bytes = await file.download_as_bytearray()
    try:
        raw_text = bytes(file_bytes).decode("utf-8", errors="ignore") if fname.endswith(".txt") \
                   else extract_pdf_text(bytes(file_bytes))
    except Exception as e:
        await update.message.reply_text(f"❌ Не вдалося прочитати файл: {e}")
        return
    if not raw_text.strip():
        await update.message.reply_text("❌ Файл не містить тексту.")
        return
    await process_transcript(update, raw_text, doc.file_name)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    urls = re.findall(r'https?://\S+', text)
    if not urls:
        return
    url = urls[0]
    await update.message.reply_text(f"⏳ Завантажую: {url}")
    try:
        raw_text = await fetch_url_text(url)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")
        return
    slug = re.sub(r'https?://', '', url).replace('/', '_')[:40]
    await process_transcript(update, raw_text, slug)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Longread conversation
    longread_handler = ConversationHandler(
        entry_points=[CommandHandler("longread", longread_start)],
        states={
            LONGREAD_WAIT_SOURCE: [
                MessageHandler(filters.Document.ALL, longread_got_source),
                MessageHandler(filters.TEXT & ~filters.COMMAND, longread_got_source),
            ],
            LONGREAD_WAIT_LANG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, longread_got_lang),
            ],
            LONGREAD_WAIT_TOPICS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, longread_got_topics),
            ],
            LONGREAD_WAIT_EDIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, longread_got_edit),
            ],
        },
        fallbacks=[CommandHandler("cancel", longread_cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("memory", show_memory))
    app.add_handler(CommandHandler("clearmemory", clear_memory))
    app.add_handler(longread_handler)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
