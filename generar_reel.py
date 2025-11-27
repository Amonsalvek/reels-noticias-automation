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
import subprocess
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
    ImageClip,
    concatenate_videoclips
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
LOGO_PATH = os.getenv("LOGO_PATH", "")  # Path to logo image file

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


def create_divider_line(target_w=1080, target_h=1920, line_height=2, color="#333333", duration=1.0):
    """Crea un clip de línea divisoria horizontal en el medio del video."""
    img = Image.new("RGB", (target_w, line_height), color)
    line_clip = ImageClip(np.array(img), ismask=False) \
        .set_duration(duration) \
        .set_position(("center", target_h // 2 - line_height // 2))
    return line_clip


def generate_srt_from_script(script_text: str, audio_duration: float, output_srt_path: Path):
    """
    Genera un archivo SRT básico desde el script.
    Divide el texto en segmentos temporales aproximados basados en la duración del audio.
    Usa una estimación de velocidad de lectura (palabras por segundo).
    """
    # Dividir el script en frases (por puntos, signos de exclamación, etc.)
    sentences = re.split(r'[.!?]+', script_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        # Si no hay frases, usar el texto completo
        sentences = [script_text]
    
    # Calcular palabras totales
    total_words = len(script_text.split())
    words_per_second = total_words / audio_duration if audio_duration > 0 else 3.0
    
    srt_content = []
    current_time = 0.0
    
    for i, sentence in enumerate(sentences):
        if not sentence:
            continue
            
        # Calcular duración basada en palabras
        sentence_words = len(sentence.split())
        sentence_duration = sentence_words / words_per_second if words_per_second > 0 else 2.0
        sentence_duration = max(1.0, min(sentence_duration, 5.0))  # Entre 1 y 5 segundos
        
        start_time = current_time
        end_time = min(current_time + sentence_duration, audio_duration)
        
        # Formatear tiempos SRT (HH:MM:SS,mmm)
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_content.append(f"{i + 1}")
        srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
        srt_content.append(sentence)
        srt_content.append("")
        
        current_time = end_time
        if current_time >= audio_duration:
            break
    
    output_srt_path.write_text("\n".join(srt_content), encoding="utf-8")


def convert_srt_to_ass(srt_path: Path, ass_path: Path, font_path: Path, video_width=1080, video_height=1920):
    """
    Convierte un archivo SRT a ASS con estilo:
    - Caja semitransparente
    - Máximo 2 líneas
    - Posicionado en la mitad superior
    """
    srt_content = srt_path.read_text(encoding="utf-8")
    
    # Parsear SRT básico
    subtitle_blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    # Obtener nombre de fuente del archivo (opcional, usar Arial como fallback)
    font_name = "Arial"
    try:
        if font_path.exists():
            try:
                from fontTools.ttLib import TTFont
                font = TTFont(str(font_path))
                font_name = font['name'].getDebugName(4) or "Arial"
            except ImportError:
                # fontTools no está instalado, usar Arial
                pass
            except Exception:
                # Error al leer la fuente, usar Arial
                pass
    except Exception:
        pass
    
    # Estilo ASS: texto blanco, borde negro, caja semitransparente
    # BorderStyle=3 = caja de fondo, Outline=4 = borde grueso, Shadow=0 = sin sombra
    # PrimaryColour=&H00FFFFFF = blanco (BGR), OutlineColour=&H00000000 = negro (BGR)
    # BackColour=&H80000000 = negro semitransparente para la caja
    ass_header = f"""[Script Info]
Title: Subtítulos
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,4,0,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    ass_events = []
    
    for block in subtitle_blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Extraer tiempo (línea 2)
        time_line = lines[1]
        time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
        if not time_match:
            continue
        
        # Convertir tiempo a formato ASS (H:MM:SS.cc)
        # Formato ASS: H:MM:SS.cc (centésimas de segundo, no milisegundos)
        def srt_to_ass_time(h, m, s, ms):
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            secs = total_seconds % 60
            # Formato: H:MM:SS.cc (2 decimales para centésimas)
            return f"{hours}:{minutes:02d}:{secs:05.2f}"
        
        start_time = srt_to_ass_time(time_match.group(1), time_match.group(2), time_match.group(3), time_match.group(4))
        end_time = srt_to_ass_time(time_match.group(5), time_match.group(6), time_match.group(7), time_match.group(8))
        
        # Texto (líneas 3+)
        text = ' '.join(lines[2:])
        # Limpiar HTML si existe
        text = re.sub(r'<[^>]+>', '', text)
        
        # Dividir en máximo 2 líneas
        words = text.split()
        if len(words) > 18:  # Aproximadamente 2 líneas
            mid = len(words) // 2
            text = ' '.join(words[:mid]) + '\\N' + ' '.join(words[mid:])
        
        # Crear evento ASS con caja semitransparente
        # Posicionado en la mitad superior, cerca de la línea divisoria (y=850, justo arriba de la línea en y=960)
        # \an5 para anclaje central (centro)
        # \pos(x,y) para posición absoluta
        # \3c&H000000& para color de borde negro
        # \4c&H000000& para color de fondo de caja
        # \4a&H99& para transparencia del fondo (~40% opacidad, más visible)
        # \bord4 para borde grueso (más visible)
        # \shad0 para sin sombra
        # \fad(200,200) para fade in/out suave
        # Texto blanco (definido en el estilo) con borde negro
        center_x = video_width // 2
        pos_y = 850  # Posición vertical, justo arriba de la línea divisoria en y=960
        ass_text = f"{{\\an5\\pos({center_x},{pos_y})\\3c&H000000&\\4c&H000000&\\4a&H99&\\bord4\\shad0\\fad(200,200)}}{text}"
        
        ass_events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{ass_text}")
    
    ass_content = ass_header + "\n".join(ass_events)
    ass_path.write_text(ass_content, encoding="utf-8")


def burn_subtitles_with_ffmpeg(video_path: Path, ass_path: Path, output_path: Path):
    """
    Quema los subtítulos ASS en el video usando FFmpeg.
    """
    # Verificar que los archivos existan
    if not video_path.exists():
        raise RuntimeError(f"Video no encontrado: {video_path}")
    if not ass_path.exists():
        raise RuntimeError(f"Archivo ASS no encontrado: {ass_path}")
    
    # Usar rutas absolutas
    video_str = str(video_path.resolve())
    ass_str = str(ass_path.resolve())
    output_str = str(output_path.resolve())
    
    # En Windows, las rutas pueden tener espacios, así que las escapamos correctamente
    # Usar el filtro ass que es más directo
    cmd = [
        "ffmpeg",
        "-i", video_str,
        "-vf", f"ass={ass_str}",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-preset", "ultrafast",
        "-y",
        output_str
    ]
    
    try:
        log(f"Quemando subtítulos: {ass_path.name}")
        if DEBUG:
            log(f"Comando FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            log(f"FFmpeg stdout: {result.stdout[:200]}")
        log(f"Subtítulos quemados exitosamente")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
        log(f"[ERROR] FFmpeg stderr: {error_msg[:500]}")
        # Mostrar el contenido del archivo ASS para debug
        if DEBUG:
            try:
                ass_content = ass_path.read_text(encoding="utf-8")
                log(f"[DEBUG] Contenido ASS (primeras 500 chars):\n{ass_content[:500]}")
            except:
                pass
        raise RuntimeError(f"Error al quemar subtítulos con FFmpeg: {error_msg[:500]}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg no encontrado. Asegúrate de que FFmpeg esté instalado y en el PATH.")


def build_video_with_overlays(
    background_video_path: Path,
    background_music_path: Path,
    font_path: Path,
    title_text: str,
    tts_audio_path: Path,
    output_path: Path,
    logo_path: Path = None,
    script_text: str = None,
):

    video = VideoFileClip(str(background_video_path)).without_audio()
    video = make_vertical_canvas(video, target_w=1080, target_h=1920)

    voice = AudioFileClip(str(tts_audio_path)).volumex(VOICE_VOLUME)
    music = AudioFileClip(str(background_music_path)).volumex(MUSIC_VOLUME)

    audio_dur = voice.duration
    final_duration = min(audio_dur - 0.05, 59.5)

    voice = voice.set_duration(final_duration)
    music_loop = afx.audio_loop(music, duration=final_duration).audio_fadeout(0.4)

    # Crop video a la mitad superior (0 a 960)
    video_cropped = video.crop(y1=0, y2=960).set_position((0, 0))
    
    # Asegurar que el video se repita hasta alcanzar final_duration
    video_duration = video_cropped.duration
    video_looped = None  # Para poder cerrarlo después si se crea
    if video_duration < final_duration:
        # Calcular cuántas veces necesitamos repetir el video
        num_loops = int(final_duration / video_duration) + 1
        # Crear lista de clips repetidos
        video_clips = [video_cropped] * num_loops
        # Concatenar todos los clips
        video_looped = concatenate_videoclips(video_clips, method="compose")
        # Recortar a la duración exacta
        video_loop = video_looped.subclip(0, final_duration)
    else:
        # Si el video es más largo, simplemente recortarlo
        video_loop = video_cropped.subclip(0, final_duration)

    # Crear fondo negro para la mitad inferior
    lower_bg = ImageClip(np.zeros((960, 1080, 3), dtype=np.uint8)) \
        .set_duration(final_duration) \
        .set_position((0, 960)) \
        .set_fps(video.fps or 30)

    # Overlay y título en la mitad inferior, justo debajo de la línea divisoria
    # La línea divisoria está en y=960, así que colocamos el overlay y título más arriba
    # Margen de 40px debajo de la línea divisoria
    overlay_y = 1000  # 960 (línea) + 40 (margen)
    overlay = create_overlay_shape().set_duration(final_duration).set_position(("center", overlay_y))

    text_clip = create_pil_text_clip(
        text=title_text,
        font_path=str(font_path),
        fontsize=80,
        color="white",
        max_width=int(1080 * 0.85),
        duration=final_duration,
        position=("center", overlay_y),  # justo debajo de la línea divisoria
    )

    # Línea divisoria horizontal en el medio
    divider_line = create_divider_line(
        target_w=1080,
        target_h=1920,
        line_height=2,
        color="#333333",
        duration=final_duration
    )

    clips_to_composite = [video_loop, lower_bg, divider_line, overlay, text_clip]

    # Coordenadas base para zona de branding (en la mitad inferior)
    base_x = 50
    base_y = 1920 - 120 - 50  # asumimos altura de logo ~120

    logo_width = 0
    logo_height = 0

    # Logo (si existe)
    if logo_path and logo_path.exists():
        logo_img = Image.open(str(logo_path))
        logo_height = 120
        aspect_ratio = logo_img.width / logo_img.height
        logo_width = int(logo_height * aspect_ratio)
        logo_img = logo_img.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

        logo_clip = (
            ImageClip(np.array(logo_img), ismask=False)
            .set_duration(final_duration)
            .set_position((base_x, base_y))
        )
        clips_to_composite.append(logo_clip)
    else:
        # Si no hay logo, dejamos un ancho mínimo de referencia para no pegar el texto al borde
        logo_width = 0
        logo_height = 0

    # Texto de branding SIEMPRE (aunque no haya logo)
    branding_x = base_x + (logo_width + 20 if logo_width > 0 else 0)

    branding_text = create_pil_text_clip(
        text="ChileTransportistas.com",
        font_path=str(font_path),
        fontsize=40,
        color="white",
        max_width=600,
        duration=final_duration,
        position=(branding_x, 0),  # luego corregimos Y
    )
    text_height = branding_text.h or 60  # fallback por si acaso
    # centrar verticalmente respecto al logo si existe; si no, usar base_y
    if logo_height > 0:
        text_y = base_y + (logo_height - text_height) // 2
    else:
        text_y = base_y + (120 - text_height) // 2

    branding_text = branding_text.set_position((branding_x, text_y))
    clips_to_composite.append(branding_text)

    final_audio = CompositeAudioClip([music_loop, voice])

    final = CompositeVideoClip(
        clips_to_composite,
        size=(1080, 1920)
    ).set_audio(final_audio)

    gc.collect()

    # Guardar video temporal antes de quemar subtítulos
    temp_video_path = output_path.parent / f"temp_{output_path.name}"
    
    final.write_videofile(
        str(temp_video_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="1500k",
        preset="ultrafast",
        threads=1,
    )

    final.close()
    video_loop.close()
    if video_looped is not None:
        video_looped.close()
    video_cropped.close()
    video.close()
    voice.close()
    music_loop.close()
    music.close()

    # Si hay script_text, generar y quemar subtítulos
    if script_text:
        try:
            log("Generando subtítulos...")
            srt_path = output_path.parent / f"{output_path.stem}.srt"
            ass_path = output_path.parent / f"{output_path.stem}.ass"
            
            generate_srt_from_script(script_text, final_duration, srt_path)
            log(f"SRT generado: {srt_path.name}")
            
            convert_srt_to_ass(srt_path, ass_path, font_path)
            log(f"ASS generado: {ass_path.name}")
            
            if DEBUG:
                # Mostrar una muestra del contenido ASS
                try:
                    ass_sample = ass_path.read_text(encoding="utf-8")[:500]
                    log(f"[DEBUG] Muestra ASS:\n{ass_sample}")
                except:
                    pass
            
            burn_subtitles_with_ffmpeg(temp_video_path, ass_path, output_path)
            
            # Limpiar archivos temporales (excepto en modo DEBUG)
            if not DEBUG:
                if temp_video_path.exists():
                    temp_video_path.unlink()
                if srt_path.exists():
                    srt_path.unlink()
                if ass_path.exists():
                    ass_path.unlink()
            else:
                log(f"[DEBUG] Archivos temporales conservados: {srt_path}, {ass_path}")
        except Exception as e:
            log(f"[WARN] Error al generar subtítulos: {e}")
            import traceback
            if DEBUG:
                log(f"[DEBUG] Traceback: {traceback.format_exc()}")
            # Si falla, usar el video sin subtítulos
            if temp_video_path.exists():
                temp_video_path.rename(output_path)
    else:
        # Si no hay script, renombrar el temporal al final
        if temp_video_path.exists():
            temp_video_path.rename(output_path)


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

        # Logo path (can be from env var or local file)
        logo_path = None
        if LOGO_PATH:
            candidate = Path(LOGO_PATH)
            # Si es relativo, asumir que está en BASE_DIR
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate
            if candidate.exists():
                logo_path = candidate
            else:
                log(f"[WARN] Logo no encontrado en ruta: {candidate}")

        # También revisar archivos locales por defecto en el directorio del script
        if not logo_path:
            for name in ["logo.png", "logo.jpg"]:
                candidate = BASE_DIR / name
                if candidate.exists():
                    logo_path = candidate
                    log(f"[INFO] Usando logo por defecto: {candidate}")
                    break

        if not logo_path:
            log("[WARN] No se encontró ningún archivo de logo. Se mostrará solo el texto de marca.")

        build_video_with_overlays(
            background_video_path=bg_video_path,
            background_music_path=bg_music_path,
            font_path=font_path,
            title_text=news_title,
            tts_audio_path=tts_path,
            output_path=output_video,
            logo_path=logo_path,
            script_text=script_text,
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