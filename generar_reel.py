#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generador de reels para UNA noticia individual.
Modo de uso:

    python generar_reel.py "https://www.sitiodenoticias.cl/articulo/12345"

Flujo:
1. Recibe una URL directa a una noticia.
2. Extrae título, fuente y URL final real.
3. Genera guion 60s con OpenAI.
4. Genera copy.
5. Genera narración TTS.
6. Descarga recursos y arma el video vertical.
7. (Opcional) Sube a Drive y envía correo.
"""

import os
import sys
import re
import tempfile
import gc
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip
)
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np


# ========================= CONFIG ========================= #

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKGROUND_VIDEO_URL = os.getenv("BACKGROUND_VIDEO_URL", "")
BACKGROUND_MUSIC_URL = os.getenv("BACKGROUND_MUSIC_URL", "")
FONT_URL = os.getenv("FONT_URL", "")

MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.4"))
VOICE_VOLUME = float(os.getenv("VOICE_VOLUME", "1.0"))

DEBUG = "--debug" in sys.argv


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# =========================================================
# EXTRAER DATOS DE LA NOTICIA DESDE UNA URL REAL
# =========================================================

def extract_news_metadata(url: str):
    """
    Extrae título, fuente y URL final de la página de la noticia.
    Usamos og:title, og:site_name y <title>.
    """

    try:
        resp = requests.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        final_url = resp.url
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar la noticia: {e}")

    # --------- TÍTULO ---------
    og_title = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.I
    )

    if og_title:
        title = og_title.group(1).strip()
    else:
        title_tag = re.search(r'<title[^>]*>(.*?)</title>', html, re.I | re.S)
        if title_tag:
            title = title_tag.group(1).strip()
            if " - " in title:
                title = title.split(" - ")[0]
        else:
            title = "Título no encontrado"

    # --------- FUENTE ---------
    og_site = re.search(
        r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.I
    )
    if og_site:
        source = og_site.group(1).strip()
    else:
        source = re.sub(r"https?://(www\.)?", "", final_url).split("/")[0]

    return title, source, final_url


# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no está configurado.")
    return OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# GENERACIÓN DE GUIÓN Y COPY
# =========================================================

def generate_script(client: OpenAI, news_title: str, source: str, final_url: str) -> str:

    prompt = f"""
Eres un guionista experto en reels informativos de máximo 60 segundos. 
La audiencia son dueños y gerentes de empresas de transporte de carga en Chile.

Escribe un guion narrado en texto plano basado únicamente en esta noticia:

Título: {news_title}
Medio: {source}
URL: {final_url}

Reglas:
- Máximo 60 segundos al leerse.
- Gancho inicial fuerte (dos frases).
- Sin mencionar la fuente ni la URL.
- Frases cortas, ritmo rápido.
- Enfocado en el impacto para el transporte de carga en Chile.
- Cierre poderoso de máximo 12 palabras.
- Sin HTML ni Markdown.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
    )

    return r.choices[0].message.content.strip()


def generate_copy(client: OpenAI, news_title: str, source: str, final_url: str) -> str:

    prompt = f"""
Escribe un COPY breve para acompañar el reel.

Reglas:
- 3 a 4 líneas.
- Lenguaje directo.
- Línea final EXACTA:
  Fuente: {source}, disponible en {final_url}
- Sin Markdown.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.6,
        messages=[{"role": "user", "content": prompt}],
    )

    return r.choices[0].message.content.strip()


# =========================================================
# TTS
# =========================================================

def generate_tts_audio(client: OpenAI, script_text: str, output_path: Path):

    instructions = (
        "Habla en español con acento chileno natural. "
        "Hombre chileno joven, directo y profesional. "
        "Narración rápida estilo reel informativo."
    )

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=script_text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(str(output_path))


# =========================================================
# PIL + MOVIEPY — OVERLAYS Y TEXTO
# =========================================================

def create_pil_text_clip(
    text: str,
    font_path: str,
    fontsize: int,
    color: str,
    max_width: int,
    duration: float,
    position=("center", "center"),
):
    font = ImageFont.truetype(font_path, fontsize)

    temp_img = Image.new("RGB", (max_width, 10), (0, 0, 0))
    draw = ImageDraw.Draw(temp_img)

    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = (current + " " + w).strip() if current else w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    padding = 20
    total_h = line_h * len(lines) + padding * 2

    img = Image.new("RGBA", (max_width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((max_width - w) / 2, y), line, font=font, fill=color)
        y += line_h

    return (
        ImageClip(np.array(img), ismask=False)
        .set_duration(duration)
        .set_position(position)
    )


def create_overlay_shape(
    target_w=1080,
    target_h=1920,
    shape_opacity=200,
    color=(220, 20, 20)
):
    H = int(target_h * 0.80)
    W = target_w

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    p1 = (int(W*0.35), 0)
    p2 = (W, 0)
    p3 = (W, int(H*0.55))
    p4 = (int(W*0.70), H)
    p5 = (0, H)
    p6 = (0, int(H*0.25))

    polygon = [p1, p2, p3, p4, p5, p6]
    draw.polygon(polygon, fill=(color[0], color[1], color[2], shape_opacity))

    rounded = img.filter(ImageFilter.GaussianBlur(3))
    return ImageClip(np.array(rounded))


def make_vertical_canvas(clip, target_w=1080, target_h=1920):
    scale = target_h / clip.h
    scaled = clip.resize(scale)

    x = (target_w - scaled.w) // 2

    bg = ImageClip(np.zeros((target_h, target_w, 3), dtype=np.uint8)) \
        .set_duration(clip.duration) \
        .set_fps(clip.fps)

    return CompositeVideoClip(
        [bg, scaled.set_position((x, 0))],
        size=(target_w, target_h)
    ).set_duration(clip.duration).set_fps(clip.fps or 30)


def build_video_with_overlays(
    background_video_path: Path,
    background_music_path: Path,
    font_path: Path,
    title_text: str,
    tts_audio_path: Path,
    output_path: Path,
):

    video = VideoFileClip(str(background_video_path)).without_audio()
    video = make_vertical_canvas(video, target_w=1080, target_h=1920)

    voice = AudioFileClip(str(tts_audio_path)).volumex(VOICE_VOLUME)
    music = AudioFileClip(str(background_music_path)).volumex(MUSIC_VOLUME)

    audio_dur = voice.duration
    final_duration = min(audio_dur - 0.05, 59.5)

    voice = voice.set_duration(final_duration)
    music_loop = afx.audio_loop(music, duration=final_duration).audio_fadeout(0.4)

    video_loop = video.fx(vfx.loop, duration=final_duration)

    overlay = create_overlay_shape().set_duration(final_duration).set_position(("center", "center"))

    text_clip = create_pil_text_clip(
        text=title_text,
        font_path=str(font_path),
        fontsize=70,
        color="white",
        max_width=int(1080 * 0.85),
        duration=final_duration,
        position=("center", "center"),
    )

    final_audio = CompositeAudioClip([music_loop, voice])

    final = CompositeVideoClip(
        [video_loop, overlay, text_clip],
        size=video_loop.size
    ).set_audio(final_audio)

    gc.collect()

    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="1500k",
        preset="ultrafast",
        threads=1,
    )

    final.close()
    video_loop.close()
    video.close()
    voice.close()
    music_loop.close()
    music.close()


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python generar_reel.py \"URL_NOTICIA\"")
        sys.exit(1)

    input_url = sys.argv[1].strip()

    log("🔍 Procesando noticia:\n" + input_url + "\n")

    news_title, source, final_url = extract_news_metadata(input_url)

    log(f"Título detectado: {news_title}")
    log(f"Fuente detectada: {source}")
    log(f"URL final: {final_url}")

    client = get_openai_client()

    video_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # carpeta temporal
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"news_video_{video_id}_"))
    if DEBUG:
        log(f"Directorio temporal: {tmp_dir}")

    try:
        # ------- generar guion -------
        script_text = generate_script(client, news_title, source, final_url)

        # ------- generar TTS -------
        tts_path = tmp_dir / f"{video_id}_audio.mp3"
        generate_tts_audio(client, script_text, tts_path)

        # ------- generar copy -------
        copy_text = generate_copy(client, news_title, source, final_url)
        (tmp_dir / f"{video_id}_copy.txt").write_text(copy_text, encoding="utf-8")

        # ------- descargar recursos -------
        bg_video_path = tmp_dir / "bg.mp4"
        bg_music_path = tmp_dir / "music.mp3"
        font_path = tmp_dir / "font.otf"

        download = lambda url, dest: requests.get(url, timeout=30).raise_for_status()
        requests.get

        r = requests.get(BACKGROUND_VIDEO_URL, stream=True)
        with open(bg_video_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)

        r = requests.get(BACKGROUND_MUSIC_URL, stream=True)
        with open(bg_music_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)

        r = requests.get(FONT_URL, stream=True)
        with open(font_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)

        # ------- generar video -------
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

    except Exception as e:
        print("❌ ERROR:", e)

    finally:
        if DEBUG:
            print("DEBUG: temporal NO borrada:", tmp_dir)
        # else:
        #     shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()