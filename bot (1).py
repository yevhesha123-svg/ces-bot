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


async def call_claude(text: str) -> str:
    """Call Anthropic API to process transcript."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 8192,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Ось сирий транскрипт подкасту. Відредагуй його за правилами:\n\n{text}"
                    }
                ]
            }
        )
        data = response.json()

        # Show exact error from Anthropic if something went wrong
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", str(data))
            raise ValueError(f"Anthropic API error {response.status_code}: {error_msg}")

        if "content" not in data:
            raise ValueError(f"Unexpected API response: {data}")

        return data["content"][0]["text"]


async def fetch_url_text(url: str) -> str:
    """Fetch text content from a URL."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        # Very basic HTML stripping
        text = re.sub(r'<[^>]+>', ' ', r.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:15000]  # limit to avoid token overflow


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Я бот ЦЕС для підготовки транскриптів подкасту.\n\n"
        "Надішли мені:\n"
        "📄 *PDF-файл* з транскриптом\n"
        "🔗 *Посилання* на сторінку з текстом (наприклад, ces.org.ua)\n\n"
        "Я відредагую текст у форматі для публікації на сайті ЦЕС.",
        parse_mode="Markdown"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("⚠️ Поки що підтримуються лише PDF-файли.")
        return

    await update.message.reply_text("⏳ Обробляю PDF, зачекайте...")

    file = await context.bot.get_file(doc.file_id)
    file_bytes = await file.download_as_bytearray()

    try:
        raw_text = extract_pdf_text(bytes(file_bytes))
    except Exception as e:
        await update.message.reply_text(f"❌ Не вдалося прочитати PDF: {e}")
        return

    if not raw_text.strip():
        await update.message.reply_text("❌ PDF не містить тексту (можливо, це скан).")
        return

    try:
        result = await call_claude(raw_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка Claude API: {e}")
        return

    # Send in chunks if too long
    await send_long_message(update, result)


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

    await update.message.reply_text("✅ Сторінку завантажено. Обробляю текст...")

    try:
        result = await call_claude(raw_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка Claude API: {e}")
        return

    await send_long_message(update, result)


async def send_long_message(update: Update, text: str):
    """Send message, splitting if over Telegram's 4096 char limit."""
    chunk_size = 4000
    if len(text) <= chunk_size:
        await update.message.reply_text(text)
        return

    parts = []
    while text:
        if len(text) <= chunk_size:
            parts.append(text)
            break
        # Try to split at newline
        split_at = text.rfind('\n', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()

    for i, part in enumerate(parts, 1):
        await update.message.reply_text(
            f"📝 Частина {i}/{len(parts)}:\n\n{part}"
        )


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
