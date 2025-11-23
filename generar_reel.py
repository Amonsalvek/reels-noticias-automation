#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generador de reels para UNA noticia individual.
Modo de uso:

    python3 generar_reel.py "https://www.sitiodenoticias.cl/articulo/12345"

Flujo:
1. Recibe una URL directa a una noticia.
2. Extrae título, fuente y URL final real.
3. Genera guion 60s con OpenAI.
4. Genera copy.
5. Genera narración TTS.
6. Descarga recursos y arma el video vertical.
7. (Opcional) Sube a Drive y envía correo.

Todo igual que antes, sin Google News.
"""

import os, sys, json, re, hashlib, tempfile, shutil, gc
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip,
    CompositeVideoClip, ImageClip, ColorClip
)
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ===================== CONFIG ===================== #

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKGROUND_VIDEO_URL = os.getenv("BACKGROUND_VIDEO_URL", "")
BACKGROUND_MUSIC_URL = os.getenv("BACKGROUND_MUSIC_URL", "")
FONT_URL = os.getenv("FONT_URL", "")

MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.4"))
VOICE_VOLUME = float(os.getenv("VOICE_VOLUME", "1.0"))

DEBUG = "--debug" in sys.argv


def log(msg):
    print(msg)
    sys.stdout.flush()


# =========================================================
# EXTRAER DATOS DE LA NOTICIA DIRECTA DESDE UNA URL
# =========================================================

def extract_news_metadata(url: str):
    """
    Extrae título, fuente y URL final real de una página de noticias.
    Usa:
    - <meta property="og:title">
    - <meta property="og:site_name">
    - <title>
    """

    try:
        resp = requests.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        final_url = resp.url
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar la noticia: {e}")

    # ---------- TÍTULO ----------
    og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if og_title:
        title = og_title.group(1).strip()
    else:
        title_tag = re.search(r'<title[^>]*>(.*?)</title>', html, re.I|re.S)
        title = title_tag.group(1).strip() if title_tag else "Título no encontrado"

        # Quitar el “ - Fuente”
        if " - " in title:
            title = title.split(" - ")[0]

    # ---------- FUENTE ----------
    og_site = re.search(r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if og_site:
        source = og_site.group(1).strip()
    else:
        # fallback → dominio
        source = re.sub(r"https?://(www\.)?", "", final_url).split("/")[0]

    return title, source, final_url


# =========================================================
# EL RESTO DEL SCRIPT QUEDA CASI IGUAL
# =========================================================

def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no está configurado.")
    return OpenAI(api_key=OPENAI_API_KEY)


def download_file(url: str, dest: Path):
    if DEBUG: log(f"[download] {url} → {dest}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

# ---------- Generación de guion y copy (igual que antes) ----------

def generate_script(client, news_title, source, final_url):
    prompt = f"""
Eres un guionista experto en reels informativos de máximo 60 segundos. 
La audiencia son dueños y gerentes de empresas de transporte de carga en Chile.

Escribe un guion narrado en texto plano basado únicamente en esta noticia:

Título: {news_title}
Medio: {source}
URL: {final_url}

Reglas:
- Máx 60 segundos cuando se lee.
- Gancho inicial fuerte (máx 2 frases).
- No mencionar fuente ni URL en el guion.
- Frases cortas, ritmo rápido.
- Enfocado en el impacto para el transporte de carga en Chile.
- Cierre poderoso de máximo 12 palabras.
- Sin Markdown, sin HTML.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()


def generate_copy(client, news_title, source, final_url):
    prompt = f"""
Escribe un copy breve para acompañar el reel.
Reglas:
- 3 a 4 líneas.
- Línea final: "Fuente: {source}, disponible en {final_url}"
- Lenguaje directo. Sin Markdown.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.6,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()


# ---------- TTS (igual que antes) ----------

def generate_tts_audio(client, script_text, output_path: Path):

    instructions = (
        "Habla en español con acento chileno natural. "
        "Hombre joven, directo y profesional. "
        "Narración estilo reel informativo, rápida pero clara."
    )

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=script_text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(str(output_path))


# =========================================================
# VIDEO BUILD (idéntico a tu versión)
# =========================================================

# ⭐️ TODO EL CÓDIGO DE MoviePy / PIL / build_video_with_overlays
# LO DEJO INTACTO PARA NO ALARGAR EL MENSAJE.
# → Copio y pego tus funciones EXACTAS aquí:

# (📌 Inserta aquí TODAS tus funciones:
#    create_pil_text_clip(), create_overlay_shape(),
#    make_vertical_canvas(), build_video_with_overlays(), etc.)
#
# No las repito en este mensaje por espacio.
#
# 👇🏻 IMPORTANTE
# Tú solo reemplaza el bloque superior por este.
# NO borres tus funciones de video.
# =========================================================



# =========================================================
# MAIN MODIFICADO
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 generar_reel.py \"URL_NOTICIA\"")
        sys.exit(1)

    input_url = sys.argv[1].strip()
    
    log(f"🔍 Procesando noticia:\n{input_url}\n")

    # 1) Extraer datos desde la URL
    news_title, source, final_url = extract_news_metadata(input_url)
    log(f"Título detectado: {news_title}")
    log(f"Fuente detectada: {source}")
    log(f"URL final: {final_url}")

    client = get_openai_client()

    video_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"news_video_{video_id}_"))
    if DEBUG:
        log(f"Directorio temporal: {tmp_dir}")

    try:
        # 2) Guion
        script_text = generate_script(client, news_title, source, final_url)
        tts_path = tmp_dir / f"{video_id}_audio.mp3"
        generate_tts_audio(client, script_text, tts_path)

        # 3) Copy
        copy_text = generate_copy(client, news_title, source, final_url)

        # 4) Descargar recursos
        bg_video_path = tmp_dir / "bg.mp4"
        bg_music_path = tmp_dir / "music.mp3"
        font_path = tmp_dir / "font.otf"

        download_file(BACKGROUND_VIDEO_URL, bg_video_path)
        download_file(BACKGROUND_MUSIC_URL, bg_music_path)
        download_file(FONT_URL, font_path)

        # 5) Construir video
        output_video = tmp_dir / f"{video_id}.mp4"

        build_video_with_overlays(
            background_video_path=bg_video_path,
            background_music_path=bg_music_path,
            font_path=font_path,
            title_text=news_title,
            tts_audio_path=tts_path,
            output_path=output_video,
        )

        print("\n✔ VIDEO GENERADO:")
        print(output_video)

        # si quieres: subir a drive + emails
        # (quedó intacto, solo se removió del flujo principal)

    finally:
        if DEBUG:
            print("DEBUG: carpeta temporal NO borrada:", tmp_dir)
        # else:
        #     shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()