#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generador de reels de noticias para ChileTransportistas.

Flujo:
1. Lee Google News (Chile) con 2 keywords: "transporte de carga" y "camiones".
2. Unifica noticias, evita duplicados con un hash guardado en news_state.json.
3. Para la primera noticia nueva:
   - Genera guion (máx ~60s) con OpenAI.
   - Genera copy breve para redes con OpenAI.
   - Genera narración TTS con OpenAI (voz onyx).
   - Descarga video de fondo, música y fuente desde URLs del .env.
   - Crea reel vertical con cuadro rojo y titular encima.
   - Mezcla música (volumen bajo) + voz.
   - Exporta video mp4 optimizado para droplet chico.
   - Sube el video a Google Drive en una carpeta propia de ese reel.
   - Envía correo con link al video y el copy.
"""

import os
import sys
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import gc

import requests
import feedparser
from dotenv import load_dotenv

from openai import OpenAI

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    ColorClip,
)
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- CONFIG BÁSICA ---------------- #

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "news_state.json"

# RSS Chile (Google News)
GOOGLE_NEWS_RSS_URLS = [
    "https://news.google.com/rss/search?q=transporte+de+carga&hl=es-419&gl=CL&ceid=CL:es-419",
    "https://news.google.com/rss/search?q=camiones&hl=es-419&gl=CL&ceid=CL:es-419",
]

# Cargar .env
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "")
GOOGLE_DRIVE_ROOT = os.getenv("GOOGLE_DRIVE_ROOT", "")

GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_PASS", "")

BACKGROUND_VIDEO_URL = os.getenv("BACKGROUND_VIDEO_URL", "")
BACKGROUND_MUSIC_URL = os.getenv("BACKGROUND_MUSIC_URL", "")
FONT_URL = os.getenv("FONT_URL", "")

MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.4"))
VOICE_VOLUME = float(os.getenv("VOICE_VOLUME", "1.0"))

DEBUG = "--debug" in sys.argv

RED_COLOR = (0xD5, 0x00, 0x32)  # #d50032


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ---------------- UTILIDADES VARIAS ---------------- #

def load_state():
    if not STATE_FILE.exists():
        return {"hashes": []}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"hashes": []}


def save_state(state: dict):
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def hash_news_content(title: str, source: str, url: str) -> str:
    """
    Crea un hash estable de la noticia para evitar duplicados.
    Se usa título + fuente + URL final.
    """
    base = f"{title.strip().lower()}|{source.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def get_final_url(url: str) -> str:
    """
    Sigue redirecciones (especialmente de Google News) para obtener la URL final.
    Si falla, devuelve la original.
    """
    try:
        resp = requests.get(url, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        return resp.url
    except Exception:
        return url


def extract_clean_title_and_source(raw_title: str):
    """
    Google News suele tener títulos tipo: "Título - Medio".
    Separamos.
    """
    parts = [p.strip() for p in raw_title.split(" - ")]
    if len(parts) >= 2:
        source = parts[-1]
        clean_title = " - ".join(parts[:-1])
    else:
        source = "Fuente desconocida"
        clean_title = raw_title
    return clean_title, source


def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no está configurado.")
    return OpenAI(api_key=OPENAI_API_KEY)


def download_file(url: str, dest: Path):
    if DEBUG:
        log(f"[download_file] Descargando {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


# ---------------- TEXTO: GUION & COPY ---------------- #

def generate_script(client: OpenAI, news_title: str, source: str, final_url: str) -> str:
    """
    Genera el guion para el reel (máx ~60s narrado).
    """
    system_prompt = (
        "Eres un guionista experto en reels informativos de máximo 60 segundos. "
        "La audiencia son dueños y gerentes de empresas de transporte de carga en Chile."
    )

    user_prompt = f"""
Eres un guionista experto en reels informativos de máximo 60 segundos. La audiencia 
son dueños y gerentes de empresas de transporte de carga en Chile. Ahora escribe un 
guion narrado en texto plano, sin títulos y sin formato, basado únicamente en esta noticia:

Título: {news_title}
Medio: {source}
URL final: {final_url}

Reglas del guion:
- Duración total: máximo 60 segundos al ser leído.
- Empieza con un gancho fuerte de máximo dos frases.
- No menciones la URL dentro del guion.
- No menciones la fuente dentro del guion.
- No agregues frases como “veamos”, “en esta noticia”, “según”, etc.
- Nada de trenes, barcos ni temas que no estén explícitamente en la noticia.
- Frases cortas. Ritmo rápido. Estilo profesional.
- Concéntrate en qué significa esto para el transporte de carga en Chile.
- Cierre: una frase final poderosa de máximo 12 palabras.
- No uses HTML ni Markdown.

Entrega solo el guion final, sin encabezados ni notas.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    script = resp.choices[0].message.content.strip()
    return script


def generate_copy(client: OpenAI, news_title: str, source: str, final_url: str) -> str:
    """
    Genera el copy breve para acompañar el reel.
    """
    user_prompt = f"""
Escribe un COPY breve para acompañar el reel en redes sociales.

Reglas:
- 3 a 4 líneas que resumen la noticia en lenguaje directo.
- Luego una línea con:
  "Fuente: {source}, disponible en {final_url}"

- No uses Markdown ni símbolos especiales.
- Devuelve solo el texto final.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un copywriter experto en contenido para redes de transporte de carga en Chile."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    copy_text = resp.choices[0].message.content.strip()
    return copy_text


# ---------------- AUDIO: TTS CON OPENAI ---------------- #

def generate_tts_audio(client: OpenAI, script_text: str, output_path: Path):
    """
    Genera audio TTS usando gpt-4o-mini-tts y voz onyx.
    """
    instructions = (
        "Habla en español con acento chileno natural. "
        "Sonido de hombre chileno joven, directo y profesional. "
        "Narración estilo reel informativo: rápido pero claro. "
        "Evita acento neutro, mexicano o colombiano. "
        "Evita melodía robótica."
    )

    if DEBUG:
        log("🎙 Generando narración con OpenAI TTS (onyx)...")

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=script_text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(str(output_path))

    if DEBUG:
        log(f"✔ Audio TTS guardado en: {output_path}")


# ---------------- PIL: TEXTO COMO IMAGECLIP ---------------- #

def create_pil_text_clip(
    text: str,
    font_path: str,
    fontsize: int,
    color: str,
    max_width: int,
    duration: float,
    position=("center", "center"),
):
    """
    Crea un ImageClip con el texto centrado, usando PIL.
    Sin ImageMagick. Compatible con Pillow 10+.
    """

    font = ImageFont.truetype(font_path, fontsize)

    # Imagen temporal para medir texto
    temp_img = Image.new("RGB", (max_width, 10), (0, 0, 0))
    draw = ImageDraw.Draw(temp_img)

    # Wrap manual de líneas
    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = (current + " " + w).strip() if current else w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    # Altura total
    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    padding = 20
    total_height = line_height * len(lines) + padding * 2

    # Imagen final transparente
    img = Image.new("RGBA", (max_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((max_width - w) / 2, y), line, font=font, fill=color)
        y += line_height

    clip = (
        ImageClip(np.array(img), ismask=False)
        .set_duration(duration)
        .set_position(position)
    )
    return clip


# ---------------- VIDEO: FORMATO REEL + OVERLAYS ---------------- #

def convert_to_reel_format(clip: VideoFileClip, target_w=720, target_h=1280) -> VideoFileClip:
    """
    Convierte cualquier video a formato vertical tipo reel (9:16),
    recortando/cropping si es necesario.
    """
    # Redimensionar por altura
    clip = clip.resize(height=target_h)
    w, h = clip.size
    if w != target_w:
        # Crop centrado a target_w
        x_center = w / 2
        x1 = int(x_center - target_w / 2)
        x2 = x1 + target_w
        clip = clip.crop(x1=x1, x2=x2, y1=0, y2=target_h)
    return clip


def build_video_with_overlays(
    background_video_path: Path,
    background_music_path: Path,
    font_path: Path,
    title_text: str,
    tts_audio_path: Path,
    output_path: Path,
):
    """
    Construye el reel:
    - Video de fondo en loop.
    - Cuadro rojo con el titular.
    - Música de fondo + voz TTS.
    """

    if DEBUG:
        log("🎬 Cargando recursos de video/audio...")

    # Video base (sin audio)
    video = VideoFileClip(str(background_video_path)).without_audio()
    video = convert_to_reel_format(video, target_w=720, target_h=1280)

    # Audio TTS (voz principal)
    voice = AudioFileClip(str(tts_audio_path)).volumex(VOICE_VOLUME)

    # Música de fondo
    music = AudioFileClip(str(background_music_path)).volumex(MUSIC_VOLUME)


    # Duración real del audio TTS
    audio_dur = voice.duration

    # Margen de seguridad (para evitar error en los últimos frames)
    safety = 0.05

    # Duración final exacta del reel
    final_duration = min(audio_dur - safety, 59.5)

    # Ajustamos audios para que NO pasen del final exacto
    voice = voice.set_duration(final_duration)
    music_loop = afx.audio_loop(music, duration=final_duration).audio_fadeout(0.4)

    # Hacemos loop del video EXACTAMENTE al largo del reel
    video_loop = video.fx(vfx.loop, duration=final_duration)


    # Mezcla de audio
    final_audio = CompositeAudioClip([music_loop, voice])

    # Cuadro rojo (fijo al centro)
    box_w = int(video_loop.w * 0.85)
    box_h = int(video_loop.h * 0.3)
    red_box = (
        ColorClip(size=(box_w, box_h), color=RED_COLOR)
        .set_duration(final_duration)
        .set_position(("center", "center"))
    )

    # Texto dentro del cuadro rojo
    text_clip = create_pil_text_clip(
        text=title_text,
        font_path=str(font_path),
        fontsize=70,
        color="white",
        max_width=int(box_w * 0.9),
        duration=final_duration,
        position=("center", "center"),
    )

    # Composición final
    final = CompositeVideoClip(
        [video_loop, red_box, text_clip],
        size=video_loop.size,
    ).set_duration(final_duration)

    final = final.set_audio(final_audio)

    if DEBUG:
        log(f"💾 Exportando video final a: {output_path}")

    # Liberar memoria antes de render
    gc.collect()

    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="1500k",
        preset="ultrafast",
        threads=1,
        verbose=DEBUG,
        logger=None if not DEBUG else "bar",
    )

    final.close()
    video.close()
    voice.close()
    music.close()
    video_loop.close()
    music_loop.close()


# ---------------- GOOGLE DRIVE ---------------- #

def get_drive_session() -> AuthorizedSession:
    if not SERVICE_ACCOUNT_FILE or not GOOGLE_DRIVE_ROOT:
        raise RuntimeError("SERVICE_ACCOUNT_FILE o GOOGLE_DRIVE_ROOT no configurados.")

    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=scopes
    )
    session = AuthorizedSession(creds)
    return session


def drive_create_folder(session: AuthorizedSession, parent_id: str, name: str) -> str:
    url = "https://www.googleapis.com/drive/v3/files"
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    resp = session.post(url, json=metadata)
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def drive_upload_file(
    session: AuthorizedSession,
    parent_id: str,
    file_path: Path,
    mime_type: str,
    name: str,
) -> str:
    """
    Sube un archivo a Drive usando uploadType=multipart.
    Devuelve fileId.
    """
    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

    boundary = "==============CHATGPTBOUNDARY=="
    metadata = {
        "name": name,
        "parents": [parent_id],
    }

    with file_path.open("rb") as f:
        file_data = f.read()

    body_parts = [
        f"--{boundary}",
        "Content-Type: application/json; charset=UTF-8",
        "",
        json.dumps(metadata),
        f"--{boundary}",
        f"Content-Type: {mime_type}",
        "",
    ]
    body_start = "\r\n".join(body_parts).encode("utf-8")
    body_end = f"\r\n--{boundary}--\r\n".encode("utf-8")

    body = body_start + file_data + body_end

    headers = {
        "Content-Type": f"multipart/related; boundary={boundary}",
    }

    resp = session.post(url, headers=headers, data=body)
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def drive_make_public(session: AuthorizedSession, file_id: str):
    """
    Marca el archivo como accesible por cualquier con el link (reader).
    """
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
    payload = {
        "role": "reader",
        "type": "anyone",
    }
    resp = session.post(url, json=payload)
    # No raise_for_status por si el scope no lo permite; no es crítico.
    if DEBUG:
        log(f"[Drive] Permisos respuesta: {resp.status_code} {resp.text[:200]}")


# ---------------- EMAIL ---------------- #

def send_email_with_video_link(
    to_email: str,
    news_title: str,
    drive_file_id: str,
    copy_text: str,
    video_id: str,
):
    """
    Envía un correo con:
    - Link al video en Drive.
    - Copy para redes.
    - Link de aprobación dummy.
    """
    if not GMAIL_USER or not GMAIL_PASS:
        log("⚠ GMAIL_USER o GMAIL_PASS no configurados; no se envía correo.")
        return

    drive_link = f"https://drive.google.com/file/d/{drive_file_id}/view"
    approve_link = f"https://chiletransportistas.com/aprobar-reel?id={video_id}"

    subject = f"[Reel noticia] {news_title}"
    body = f"""
Se generó un nuevo reel basado en la noticia:

Título:
{news_title}

Link al video en Google Drive:
{drive_link}

Copy sugerido para redes:
--------------------------------
{copy_text}
--------------------------------

Link de aprobación (dummy):
{approve_link}

Slds,
Bot de reels ChileTransportistas
"""

    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)

    log(f"📧 Correo enviado a {to_email}")


# ---------------- MAIN ---------------- #

def fetch_all_news_entries():
    """
    Lee todos los RSS de la lista GOOGLE_NEWS_RSS_URLS y unifica sus entries.
    """
    entries = []
    for url in GOOGLE_NEWS_RSS_URLS:
        if DEBUG:
            log(f"Descargando RSS: {url}")
        feed = feedparser.parse(url)
        if feed.bozo:
            log(f"⚠ Error al parsear feed: {url}")
            continue
        entries.extend(feed.entries)
    return entries


def main():
    if DEBUG:
        log("Descargando RSS de Google News (Chile)...")

    client = get_openai_client()

    # Estado de noticias ya procesadas
    state = load_state()
    seen_hashes = set(state.get("hashes", []))

    # Unificar todas las noticias
    entries = fetch_all_news_entries()
    if not entries:
        log("No se encontraron entradas en los feeds.")
        return

    # Ordenar por fecha (si existe)
    def entry_key(e):
        return getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None) or (0,)

    entries_sorted = sorted(entries, key=entry_key, reverse=True)

    new_entry = None
    entry_hash = None
    final_url = None

    for entry in entries_sorted:
        raw_title = entry.title
        clean_title, source = extract_clean_title_and_source(raw_title)
        entry_url = entry.link
        final_url_candidate = get_final_url(entry_url)
        h = hash_news_content(clean_title, source, final_url_candidate)
        if h not in seen_hashes:
            new_entry = entry
            entry_hash = h
            final_url = final_url_candidate
            break

    if not new_entry:
        log("No hay noticias nuevas por procesar.")
        return

    raw_title = new_entry.title
    clean_title, source = extract_clean_title_and_source(raw_title)
    news_title = clean_title
    news_url = final_url

    video_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if DEBUG:
        log(f"Nueva noticia encontrada: {raw_title}")
        log(f"ID de video: {video_id}")

    # Directorio temporal para este reel
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"video_news_{video_id}_"))
    if DEBUG:
        log(f"Directorio temporal: {tmp_dir}")

    try:
        # 1) Generar guion
        script_text = generate_script(client, news_title, source, news_url)
        script_path = tmp_dir / f"{video_id}_guion.txt"
        script_path.write_text(script_text, encoding="utf-8")
        if DEBUG:
            log(f"Guion guardado en: {script_path}")

        # 2) Generar copy
        copy_text = generate_copy(client, news_title, source, news_url)
        copy_path = tmp_dir / f"{video_id}_copy.txt"
        copy_path.write_text(copy_text, encoding="utf-8")
        if DEBUG:
            log(f"Copy guardado en: {copy_path}")

        # 3) Generar audio TTS
        tts_path = tmp_dir / f"{video_id}_audio.mp3"
        generate_tts_audio(client, script_text, tts_path)

        # 4) Descargar recursos (video, música, fuente)
        bg_video_path = tmp_dir / "background_video.mp4"
        bg_music_path = tmp_dir / "background_music.mp3"
        font_path = tmp_dir / "font.otf"

        download_file(BACKGROUND_VIDEO_URL, bg_video_path)
        download_file(BACKGROUND_MUSIC_URL, bg_music_path)
        download_file(FONT_URL, font_path)

        # 5) Construir video final
        output_video_path = tmp_dir / f"{video_id}_video.mp4"
        build_video_with_overlays(
            background_video_path=bg_video_path,
            background_music_path=bg_music_path,
            font_path=font_path,
            title_text=news_title,
            tts_audio_path=tts_path,
            output_path=output_video_path,
        )

        print("DEBUG: Me detengo aquí, no subo a Drive.")
        print("Ruta del video:", final_video_path)

        return   # ⬅️ DETIENE EL PROGRAMA COMPLETO

        # 6) Subir a Google Drive
        session = get_drive_session()
        # Carpeta para este reel
        safe_title = re.sub(r"[^\w\s\-]", "", news_title).strip()
        folder_name = f"{video_id} - {safe_title[:80]}"
        folder_id = drive_create_folder(session, GOOGLE_DRIVE_ROOT, folder_name)

        file_id = drive_upload_file(
            session=session,
            parent_id=folder_id,
            file_path=output_video_path,
            mime_type="video/mp4",
            name=f"{video_id}.mp4",
        )

        drive_make_public(session, file_id)

        # 7) Enviar correo
        send_email_with_video_link(
            to_email=GMAIL_USER or "amonsalvek@gmail.com",
            news_title=news_title,
            drive_file_id=file_id,
            copy_text=copy_text,
            video_id=video_id,
        )

        # 8) Actualizar estado
        seen_hashes.add(entry_hash)
        state["hashes"] = list(seen_hashes)[-1000:]
        save_state(state)

    finally:
        # Limpieza de temporales
        if DEBUG:
            log(f"Limpiando temporales en {tmp_dir}")
        print("DEBUG: No se borró la carpeta temporal:", tmpdir)
        # shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
