import os
import re
import logging
import httpx
import pdfplumber
from io import BytesIO
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

SYSTEM_PROMPT = """Ти — асистент подкасту «Що з економікою?» Центру економічної стратегії (ЦЕС).

Твоє завдання — перетворювати сирий транскрипт подкасту на відредаговану текстову розшифровку для публікації на сайті ЦЕС.

ПРАВИЛА ФОРМАТУВАННЯ:
1. Кожен спікер виділяється жирним: **Ім'я Прізвище:**
2. Типові спікери: Ангеліна Завадецька, Максим Самойлюк — і гість епізоду (його ім'я визнач з контексту)
3. Прибирай прев'ю/тизер на початку — транскрипт починається з привітання Ангеліни
4. Виправляй граматичні та стилістичні помилки автотранскрибації
5. Виправляй помилкові назви (наприклад, «Румська протока» → «Ормузька протока»)
6. Прибирай заминки, повтори, слова-паразити — але зберігай суть і стиль мовлення
7. Правильно розподіляй репліки між спікерами (транскрибатор часто плутає)
8. Зберігай природний розмовний стиль — не перетворюй на офіційний текст
9. НЕ додавай нічого від себе — лише те, що є в оригіналі
10. Починай одразу з тексту, без вступних коментарів

СТРУКТУРА ВИВОДУ:
**Ім'я Прізвище:** Текст репліки

**Ім'я Прізвище:** Текст репліки

і так далі."""

CHUNK_SIZE = 15000
# How many chars to accumulate before updating the live message
STREAM_UPDATE_EVERY = 300


async def call_claude_streaming(text: str, on_chunk) -> str:
    """Call Anthropic API with streaming. Calls on_chunk(delta) for each text piece."""
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
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Ось сирий транскрипт подкасту. Відредагуй його за правилами:\n\n{text}"
                    }
                ]
            }
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                data = __import__('json').loads(body)
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
                    event = __import__('json').loads(payload)
                except Exception:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {}).get("text", "")
                    if delta:
                        full_text += delta
                        await on_chunk(delta)

            return full_text


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
        text = re.sub(r'<[^>]+>', ' ', r.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def extract_pdf_text(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


async def send_as_file(update: Update, text: str, filename: str):
    """Send final result as a downloadable .txt file."""
    file_bytes = text.encode("utf-8")
    await update.message.reply_document(
        document=BytesIO(file_bytes),
        filename=filename,
        caption="✅ Готово! Відкрийте файл і скопіюйте текст на сайт ЦЕС."
    )


async def process_and_send(update: Update, raw_text: str, source_name: str = "transcript"):
    """Process text with streaming — user sees live progress, gets file at the end."""
    chunks = split_into_chunks(raw_text)
    total = len(chunks)
    all_results = []

    if total > 1:
        await update.message.reply_text(
            f"📋 Текст великий — розбиваю на {total} частини і обробляю кожну окремо..."
        )

    for i, chunk in enumerate(chunks, 1):
        label = f"частина {i}/{total}" if total > 1 else "транскрипт"

        # Send initial "thinking" message
        live_msg = await update.message.reply_text(
            f"✍️ Редагую {label}...\n\n_(текст з'явиться тут)_",
            parse_mode="Markdown"
        )

        accumulated = ""   # buffer for live updates
        full_result = ""   # complete result for this chunk
        chars_since_update = 0

        async def on_chunk(delta: str):
            nonlocal accumulated, full_result, chars_since_update
            full_result += delta
            accumulated += delta
            chars_since_update += len(delta)

            # Update live message every STREAM_UPDATE_EVERY chars
            if chars_since_update >= STREAM_UPDATE_EVERY:
                preview = full_result[-1500:] if len(full_result) > 1500 else full_result
                try:
                    await live_msg.edit_text(
                        f"✍️ Редагую {label}...\n\n{preview}",
                    )
                except Exception:
                    pass  # ignore edit errors (e.g. message not modified)
                chars_since_update = 0

        try:
            full_result = await call_claude_streaming(chunk, on_chunk)
            all_results.append(full_result)
        except Exception as e:
            await update.message.reply_text(
                f"❌ Помилка на {label}: {type(e).__name__}: {e}"
            )
            return

        # Final update for this chunk
        try:
            preview = full_result[-1500:] if len(full_result) > 1500 else full_result
            await live_msg.edit_text(f"✅ {label.capitalize()} готова!\n\n{preview}")
        except Exception:
            pass

    # Send everything as one file
    full_text = "\n\n".join(all_results)
    filename = re.sub(r'[^\w\-.]', '_', source_name.replace(".pdf", "").replace(".txt", "")) + "_відредаговано.txt"
    await send_as_file(update, full_text, filename)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Я бот ЦЕС для підготовки транскриптів подкасту.\n\n"
        "Надішли мені:\n"
        "📄 PDF або TXT файл з транскриптом\n"
        "🔗 Посилання на сторінку з текстом (наприклад, ces.org.ua)\n\n"
        "Я відредагую текст у реальному часі і поверну готовий файл для публікації на сайті ЦЕС."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    fname = doc.file_name.lower()

    if not fname.endswith(".pdf") and not fname.endswith(".txt"):
        await update.message.reply_text("⚠️ Надішліть PDF або TXT файл з транскриптом.")
        return

    await update.message.reply_text("⏳ Читаю файл...")

    file = await context.bot.get_file(doc.file_id)
    file_bytes = await file.download_as_bytearray()

    try:
        if fname.endswith(".txt"):
            raw_text = bytes(file_bytes).decode("utf-8", errors="ignore")
        else:
            raw_text = extract_pdf_text(bytes(file_bytes))
    except Exception as e:
        await update.message.reply_text(f"❌ Не вдалося прочитати файл: {e}")
        return

    if not raw_text.strip():
        await update.message.reply_text("❌ Файл не містить тексту.")
        return

    await process_and_send(update, raw_text, source_name=doc.file_name)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    urls = re.findall(r'https?://\S+', text)
    if not urls:
        return

    url = urls[0]
    await update.message.reply_text(f"⏳ Завантажую сторінку: {url}")

    try:
        raw_text = await fetch_url_text(url)
    except Exception as e:
        await update.message.reply_text(f"❌ Не вдалося завантажити сторінку: {e}")
        return

    await update.message.reply_text("✅ Завантажено. Обробляю...")
    slug = re.sub(r'https?://', '', url).replace('/', '_')[:40]
    await process_and_send(update, raw_text, source_name=slug)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), handle_url))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
