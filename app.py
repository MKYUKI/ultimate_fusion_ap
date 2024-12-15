import streamlit as st
import os
import random
import re
from io import BytesIO
import requests
import numpy as np
import pandas as pd
from PIL import Image
import exifread
import pdfplumber
from docx import Document
from google.oauth2 import service_account
from google.cloud import texttospeech
import plotly.express as px
import plotly.graph_objs as go
import streamlit.components.v1 as components
from pydub import AudioSegment
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from googleapiclient.discovery import build
import bcrypt
import tempfile
import json
import seaborn as sns
import matplotlib.pyplot as plt
import database  # データベースモジュールをインポート
import time
from streamlit_lottie import st_lottie
from functools import lru_cache  # キャッシュ用ライブラリ
import gettext  # 多言語対応用ライブラリ

# -----------------------------------
# 1. 初期設定
# -----------------------------------
st.set_page_config(page_title="融合アプリ", page_icon="✨", layout="wide")

# 多言語対応設定
# サポートする言語のリスト
LANGUAGES = {
    "日本語": "ja",
    "English": "en",
    "Español": "es",  # スペイン語の例
    # 必要に応じて他の言語を追加
}

# デフォルト言語設定
DEFAULT_LANGUAGE = "日本語"

# 翻訳辞書の作成
translations = {
    "ja": {
        "Welcome": "ようこそ",
        "Login": "ログイン",
        "Logout": "ログアウト",
        "Register": "登録",
        "Username": "ユーザー名",
        "Password": "パスワード",
        "Confirm Password": "パスワード確認",
        "Name": "名前",
        "Email": "メールアドレス",
        "Submit": "送信",
        "Feedback": "フィードバック",
        "Send": "送信",
        "File Upload": "ファイルアップロード",
        "Upload": "アップロード",
        "Download": "ダウンロード",
        "Settings": "設定",
        "Profile": "プロフィール",
        "Activity Log": "アクティビティログ",
        "Dashboard": "ダッシュボード",
        "Select Language": "言語選択",
        "Theme Selection": "テーマ選択",
        # 他の必要なテキストを追加
    },
    "en": {
        "Welcome": "Welcome",
        "Login": "Login",
        "Logout": "Logout",
        "Register": "Register",
        "Username": "Username",
        "Password": "Password",
        "Confirm Password": "Confirm Password",
        "Name": "Name",
        "Email": "Email",
        "Submit": "Submit",
        "Feedback": "Feedback",
        "Send": "Send",
        "File Upload": "File Upload",
        "Upload": "Upload",
        "Download": "Download",
        "Settings": "Settings",
        "Profile": "Profile",
        "Activity Log": "Activity Log",
        "Dashboard": "Dashboard",
        "Select Language": "Select Language",
        "Theme Selection": "Theme Selection",
        # 他の必要なテキストを追加
    },
    "es": {
        "Welcome": "Bienvenido",
        "Login": "Iniciar sesión",
        "Logout": "Cerrar sesión",
        "Register": "Registrarse",
        "Username": "Nombre de usuario",
        "Password": "Contraseña",
        "Confirm Password": "Confirmar contraseña",
        "Name": "Nombre",
        "Email": "Correo electrónico",
        "Submit": "Enviar",
        "Feedback": "Retroalimentación",
        "Send": "Enviar",
        "File Upload": "Carga de archivos",
        "Upload": "Subir",
        "Download": "Descargar",
        "Settings": "Configuraciones",
        "Profile": "Perfil",
        "Activity Log": "Registro de actividad",
        "Dashboard": "Tablero",
        "Select Language": "Seleccionar idioma",
        "Theme Selection": "Selección de tema",
        # 他の必要なテキストを追加
    },
    # 他の言語の翻訳を追加
}

# 翻訳関数
def _(text):
    lang = st.session_state.get("language", "ja")
    return translations.get(lang, translations["ja"]).get(text, text)

# Google Cloud TTS認証
if "gcp_service_account" in st.secrets:
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    st.session_state["tts_client"] = tts_client
else:
    st.error(_("Google Cloudサービスアカウント情報がst.secretsにありません。設定してください。"))
    st.stop()

# OpenAI APIキー設定
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    openai.api_key = None

# YouTube APIキー設定
if "youtube" in st.secrets and "api_key" in st.secrets["youtube"]:
    youtube_api_key = st.secrets["youtube"]["api_key"]
else:
    youtube_api_key = None

# セッション状態の初期化
if "messages" not in st.session_state:
    welcome_messages = [
        _("Hello! I'm Exifa, an AI assistant designed to make image metadata meaningful. Ask me anything!"),
        _("Hi! I'm Exifa, an AI-powered assistant for extracting and explaining EXIF data. How can I help you today?"),
        _("Hey! I'm Exifa, your AI-powered guide to understanding the metadata in your images. What would you like to explore?"),
        _("Hi there! I'm Exifa, an AI-powered tool built to help you make sense of your image metadata. How can I help you today?"),
        _("Hello! I'm Exifa, an AI-driven tool designed to help you understand your images' metadata. What can I do for you?"),
        _("Hi! I'm Exifa, an AI-driven assistant designed to make EXIF data easy to understand. How can I help you today?"),
        _("Welcome! I'm Exifa, an intelligent AI-powered tool for extracting and explaining EXIF data. How can I assist you today?"),
        _("Hello! I'm Exifa, your AI-powered guide for understanding image metadata. Ask me anything!"),
        _("Hi! I'm Exifa, an intelligent AI assistant ready to help you understand your images' metadata. What would you like to explore?"),
        _("Hey! I'm Exifa, an AI assistant for extracting and explaining EXIF data. How can I help you today?"),
    ]
    message = random.choice(welcome_messages)
    st.session_state["messages"] = [{"role": "assistant", "content": message}]
if "exif_df" not in st.session_state:
    st.session_state["exif_df"] = pd.DataFrame()
if "url_exif_df" not in st.session_state:
    st.session_state["url_exif_df"] = pd.DataFrame()
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "follow_up" not in st.session_state:
    st.session_state.follow_up = False
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True
if "theme" not in st.session_state:
    st.session_state["theme"] = "ダークモード"  # デフォルトをダークモードに設定
if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE  # デフォルト言語を設定

# -----------------------------------
# 2. 幻想的粒子アニメーション背景
# -----------------------------------
particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

components.html(particles_js, height=0, width=0)

# -----------------------------------
# 3. ユーティリティ関数群
# -----------------------------------

@st.cache_data(ttl=600)  # キャッシュの有効期限を10分に設定
def load_image_cached(file):
    return Image.open(file)

@st.cache_data(ttl=600)  # キャッシュの有効期限を10分に設定
def fetch_github_repos(username):
    url = f'https://api.github.com/users/{username}/repos'
    params = {
        'sort': 'updated',
        'per_page': 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return []

@st.cache_data(ttl=600)  # キャッシュの有効期限を10分に設定
def fetch_youtube_videos(channel_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=5,
        order="date"
    )
    response = request.execute()
    return response.get('items', [])

def clear_url():
    st.session_state["image_url"] = ""

def clear_files():
    st.session_state["uploaded_files"] = None
    st.session_state["file_uploader_key"] = not st.session_state.get("file_uploader_key", False)

def clear_chat_history():
    st.session_state["messages"] = [{"role": "assistant", "content": random.choice([
        _("ここは人類史上初の究極融合アプリ。チャット履歴をクリアしました。再び新たなる時代へ踏み出そう。")
    ])}]
    st.session_state["exif_df"] = pd.DataFrame()
    st.session_state["url_exif_df"] = pd.DataFrame()
    st.session_state["uploaded_files"] = None
    st.session_state["image_url"] = ""
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    elif hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    st.success(_("チャット履歴をクリアしました！"))

def load_image(file):
    if isinstance(file, str):
        response = requests.get(file)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    elif isinstance(file, bytes):
        return Image.open(BytesIO(file))
    else:
        return Image.open(file)

def clear_exif_data(image_input):
    if isinstance(image_input, BytesIO):
        image_input.seek(0)
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError(_("画像タイプがサポートされていません"))
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    buffered = BytesIO()
    image_without_exif.save(buffered, format="JPEG", quality=100, optimize=True)
    buffered.seek(0)
    return buffered.getvalue()

def download_image(data):
    st.download_button(
        label=_("⇩ EXIF除去後の画像ダウンロード"),
        data=data,
        file_name="image_no_exif.jpg",
        mime="image/jpeg",
    )

def detect_language(text):
    if re.search('[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja-JP'
    return 'en-US'

def synthesize_speech_chunk(text, lang_code, gender='neutral'):
    max_chars = 4500
    chunks = [text[i:i+max_chars] for i in range(0,len(text),max_chars)]

    gender_map = {
        'default': texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED,
        'male': texttospeech.SsmlVoiceGender.MALE,
        'female': texttospeech.SsmlVoiceGender.FEMALE,
        'neutral': texttospeech.SsmlVoiceGender.NEUTRAL
    }

    combined_audio = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        synthesis_input = texttospeech.SynthesisInput(text=chunk)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            ssml_gender=gender_map.get(gender, texttospeech.SsmlVoiceGender.NEUTRAL)
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        segment = AudioSegment.from_file(BytesIO(response.audio_content), format="mp3")
        combined_audio += segment

    output_buffer = BytesIO()
    combined_audio.export(output_buffer, format="mp3")
    output_buffer.seek(0)
    return output_buffer

def classify_image(image_path, replicate_api_token):
    """
    Replicate APIを使用して画像を分類します。
    
    Args:
        image_path (str): 画像ファイルのパス。
        replicate_api_token (str): Replicate APIトークン。
    
    Returns:
        dict: 分類結果。
    """
    model_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {replicate_api_token}",
        "Content-Type": "application/json",
    }
    
    # 使用するモデルの詳細（例: Image Classificationモデル）
    model_version = "YOUR_SELECTED_MODEL_VERSION_ID"  # ここに実際のモデルバージョンIDを入力
    
    data = {
        "version": model_version,
        "input": {
            "image": f"@{image_path}"
        }
    }
    
    response = requests.post(model_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        prediction = response.json()
        return prediction
    else:
        st.error(_("画像分類に失敗しました。ステータスコード: ") + str(response.status_code))
        st.error(response.text)
        return None

def generate_gpt_response(messages, db):
    if not openai.api_key:
        return _("OpenAI APIキーが未設定です。Secretsで設定してください。")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"{_('エラーが発生しました')}: {str(e)}"

def generate_gpt_response_stream(messages, db):
    if not openai.api_key:
        yield _("OpenAI APIキーが未設定です。Secretsで設定してください。")
        return
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True  # ストリーミングを有効化
        )
        full_response = ""
        for chunk in response:
            if 'choices' in chunk:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    full_response += delta['content']
                    yield delta['content']
    except Exception as e:
        yield f"{_('エラーが発生しました')}: {str(e)}"

def is_admin(username, db):
    return database.is_admin_user(username, db)

def admin_feedback_view(db):
    st.header(_("管理者用フィードバック閲覧"))
    feedbacks = database.get_all_feedback(db)
    if not feedbacks:
        st.info(_("まだフィードバックがありません。"))
    else:
        for fb in feedbacks:
            user = db.query(database.User).filter_by(id=fb.user_id).first()
            user_name = user.name if user else _("不明なユーザー")
            st.markdown(f"**{_('ユーザー名')}:** {user_name}")
            st.markdown(f"**{_('日時')}:** {fb.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**{_('フィードバック')}:** {fb.feedback}")
            st.markdown("---")

def admin_image_classification_view(db):
    st.header(_("管理者用画像分類結果閲覧"))
    classifications = database.get_all_image_classifications(db)
    if not classifications:
        st.info(_("まだ画像分類結果がありません。"))
    else:
        for cls in classifications:
            user = db.query(database.User).filter_by(id=cls.user_id).first()
            user_name = user.name if user else _("不明なユーザー")
            st.markdown(f"**{_('ユーザー名')}:** {user_name}")
            st.markdown(f"**{_('日時')}:** {cls.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**{_('画像')}:**")
            st.image(cls.image_path, width=200)
            st.markdown(f"**{_('分類結果')}:** {cls.classification_result}")
            st.markdown("---")

def get_exif_statistics(exif_df):
    """
    EXIFデータから統計情報を抽出します。
    
    Args:
        exif_df (pd.DataFrame): EXIFデータのデータフレーム。
    
    Returns:
        dict: 統計情報。
    """
    stats = {}
    
    # カメラモデルのカウント
    if 'Model' in exif_df.columns:
        stats[_('カメラモデルの種類')] = exif_df['Model'].value_counts().to_dict()
    
    # 露出時間の平均
    if 'ExposureTime' in exif_df.columns:
        try:
            exposure_times = exif_df['ExposureTime'].apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1]) if '/' in x else float(x))
            stats[_('露出時間の平均 (秒)')] = exposure_times.mean()
        except:
            stats[_('露出時間の平均 (秒)')] = _("計算不能")
    
    # ISO感度の平均
    if 'ISOSpeedRatings' in exif_df.columns:
        try:
            iso_values = exif_df['ISOSpeedRatings'].astype(float)
            stats[_('ISO感度の平均')] = iso_values.mean()
        except:
            stats[_('ISO感度の平均')] = _("計算不能")
    
    # 焦点距離の平均
    if 'FocalLength' in exif_df.columns:
        try:
            focal_lengths = exif_df['FocalLength'].apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1]) if '/' in x else float(x))
            stats[_('焦点距離の平均 (mm)')] = focal_lengths.mean()
        except:
            stats[_('焦点距離の平均 (mm)')] = _("計算不能")
    
    return stats

def plot_exif_statistics(exif_df):
    """
    EXIFデータの統計情報をプロットします。
    
    Args:
        exif_df (pd.DataFrame): EXIFデータのデータフレーム。
    """
    st.markdown(_("### EXIFデータの統計情報"))
    stats = get_exif_statistics(exif_df)
    
    for key, value in stats.items():
        st.markdown(f"**{key}:**")
        if isinstance(value, dict):
            # バーチャートで表示
            labels = list(value.keys())
            counts = list(value.values())
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=counts, y=labels, ax=ax, palette="viridis")
            ax.set_xlabel(_('数'))
            ax.set_ylabel('')
            ax.set_title(key)
            st.pyplot(fig)
        elif isinstance(value, float) or isinstance(value, int):
            st.write(f"{value:.2f}")
        else:
            st.write(value)
    
    # 露出時間の分布
    if 'ExposureTime' in exif_df.columns:
        try:
            exposure_times = exif_df['ExposureTime'].apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1]) if '/' in x else float(x))
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(exposure_times, bins=20, kde=True, ax=ax, color='skyblue')
            ax.set_title(_('露出時間の分布'))
            ax.set_xlabel(_('露出時間 (秒)'))
            st.pyplot(fig)
        except:
            pass
    
    # ISO感度の分布
    if 'ISOSpeedRatings' in exif_df.columns:
        try:
            iso_values = exif_df['ISOSpeedRatings'].astype(float)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(iso_values, bins=20, kde=True, ax=ax, color='salmon')
            ax.set_title(_('ISO感度の分布'))
            ax.set_xlabel(_('ISO感度'))
            st.pyplot(fig)
        except:
            pass

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# LottieアニメーションのURL
success_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")  # 成功通知
error_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")    # エラー通知
info_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x62chJ.json")     # 情報通知

def notify(message: str, type: str = "info"):
    if type == "success":
        st_lottie(success_lottie, height=100, key="success")
        st.success(message)
    elif type == "error":
        st_lottie(error_lottie, height=100, key="error")
        st.error(message)
    else:
        st_lottie(info_lottie, height=100, key="info")
        st.info(message)

# -----------------------------------
# 4. サイドバーの構築（第1回目から第12回目まで統合）
# -----------------------------------
with st.sidebar:
    # 言語選択
    st.sidebar.markdown("---")
    st.sidebar.markdown("### " + _("Select Language"))
    language = st.sidebar.selectbox("", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(st.session_state["language"]))
    st.session_state["language"] = language

    # ユーザー登録フォームを表示
    def register(db):
        st.sidebar.header(_("Register"))
        with st.sidebar.form("registration_form"):
            new_username = st.text_input(_("Username"))
            new_name = st.text_input(_("Name"))
            new_email = st.text_input(_("Email"))
            new_password = st.text_input(_("Password"), type="password")
            confirm_password = st.text_input(_("Confirm Password"), type="password")
            submitted = st.form_submit_button(_("Submit"))
            
            if submitted:
                if new_password != confirm_password:
                    st.sidebar.error(_("Passwords do not match."))
                    notify(_("Passwords do not match."), type="error")
                elif database.user_exists(new_username, db):
                    st.sidebar.error(_("This username already exists. Please choose another one."))
                    notify(_("This username already exists. Please choose another one."), type="error")
                elif new_email and database.email_exists(new_email, db):
                    st.sidebar.error(_("This email address is already registered. Please use another email address."))
                    notify(_("This email address is already registered. Please use another email address."), type="error")
                elif not new_username or not new_name or not new_password or not new_email:
                    st.sidebar.error(_("Please fill in all fields."))
                    notify(_("Please fill in all fields."), type="error")
                else:
                    database.add_user(new_username, new_name, new_password, new_email, db)
                    st.sidebar.success(_("Registration completed. Please log in."))
                    notify(_("Registration completed. Please log in."), type="success")

    # 認証システムの初期化
    def load_config_app():
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
        return config

    config_app = load_config_app()

    # データベースセッションの取得
    db = next(database.get_db())

    credentials = {
        "usernames": {
            user.username: {
                "name": user.name,
                "password": user.password  # 既にハッシュ化されたパスワード
            } for user in db.query(database.User).all()
        }
    }

    authenticator = stauth.Authenticate(
        credentials,
        config_app['cookie']['name'],
        config_app['cookie']['key'],
        config_app['cookie']['expiry_days'],
        config_app['preauthorized']
    )

    st.session_state.authenticator = authenticator

    # ユーザー登録フォームを表示
    register(db)

    # ログインフォームを表示
    name, authentication_status, username = st.session_state.authenticator.login(_("Login"), "main")

    if authentication_status:
        st.sidebar.success(f"{_('Welcome')} *{name}* {_('!')}")
        st.session_state["username"] = username
        st.session_state["name"] = name
        st.session_state.authenticator.logout(_("Logout"), "sidebar")
        
        # ユーザープロフィールの表示
        st.sidebar.markdown("### " + _("Profile"))
        st.sidebar.markdown(f"**{_('Name')}:** {name}")
        st.sidebar.markdown(f"**{_('Username')}:** {username}")
        user = db.query(database.User).filter_by(username=username).first()
        if user and user.email:
            st.sidebar.markdown(f"**{_('Email')}:** {user.email}")
        
        # プロフィールリンクの表示
        st.sidebar.markdown("""
            [GitHub](https://github.com/MKYUKI) |
            [YouTube](https://www.youtube.com/@mk_agi) |
            [Paypal](https://www.paypal.com/paypalme/MasakiKusaka) |
            [Dropbox](https://www.dropbox.com/home) |
            [HuggingFace](https://huggingface.co/pricing) |
            [X](https://x.com/MK_ASI1) |
            [Facebook](https://www.facebook.com/) |
            [Amazon JP](https://www.amazon.co.jp/s?i=digital-text&rh=p_27%3AMasaki+Kusaka&s=relevancerank&text=Masaki+Kusaka&ref=dp_byline_sr_ebooks_1) |
            [Amazon US](https://www.amazon.com/s?i=digital-text&rh=p_27%3AMasaki+Kusaka&s=relevancerank&text=Masaki+Kusaka&ref=dp_byline_sr_ebooks_1)
        """)

        # フィードバックフォーム
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + _("Feedback"))
        with st.sidebar.form("feedback_form"):
            feedback = st.text_area(_("Please provide your feedback or comments here."), "")
            submit_feedback = st.form_submit_button(_("Send"))
        if submit_feedback and feedback:
            user_obj = db.query(database.User).filter_by(username=username).first()
            if user_obj:
                database.add_feedback(user_obj.id, feedback, db)
                database.add_activity_log(user_obj.id, _("Feedback submitted."), db)
                st.sidebar.success(_("Thank you for your feedback!"))
                notify(_("Feedback has been successfully submitted."), type="success")
            else:
                st.sidebar.error(_("Failed to submit feedback. Please log in again."))
                notify(_("Failed to submit feedback. Please log in again."), type="error")
        elif submit_feedback:
            st.sidebar.warning(_("Please enter your feedback before submitting."))
            notify(_("Please enter your feedback before submitting."), type="info")
        
        # ソーシャルメディア連携の表示
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + _("Latest GitHub Repositories"))
        github_username = username  # ユーザー名がGitHubのユーザー名と一致する場合
        repos = fetch_github_repos(github_username)
        if repos:
            for repo in repos:
                repo_name = repo['name']
                repo_url = repo['html_url']
                repo_description = repo['description'] or _("No description")
                st.sidebar.markdown(f"- [{repo_name}]({repo_url}): {repo_description}")
        else:
            st.sidebar.info(_("No repositories found."))
        
        if youtube_api_key:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### " + _("Latest YouTube Videos"))
            youtube_channel_id = "YOUR_YOUTUBE_CHANNEL_ID"  # ユーザーのYouTubeチャンネルIDに置き換えてください
            videos = fetch_youtube_videos(youtube_channel_id, youtube_api_key)
            if videos:
                for video in videos:
                    video_title = video['snippet']['title']
                    video_id = video['id']['videoId']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    st.sidebar.markdown(f"- [{video_title}]({video_url})")
            else:
                st.sidebar.info(_("No videos found."))
        else:
            st.sidebar.info(_("YouTube API key is not set."))
        
        # 画像分類機能の表示
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + _("Image Classification"))
        if "REPLICATE_API_TOKEN" in st.secrets:
            replicate_api = st.secrets["REPLICATE_API_TOKEN"]
        else:
            replicate_api = st.text_input(_("Enter your Replicate API Token:"), type="password")
            if not (replicate_api.startswith("r8_") and len(replicate_api) == 40):
                st.warning(_("Please enter a valid Replicate API token."), icon="⚠️")
                st.markdown(
                    f"**{_('If you do not have an API token, please obtain one from [Replicate](https://replicate.com/account/api-tokens).')}**"
                )
        os.environ["REPLICATE_API_TOKEN"] = replicate_api
        st.markdown(_("You can upload an image to classify it using AI if you have set your Replicate API token."))
        
        # テーマ切り替え
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + _("Theme Selection"))
        theme = st.sidebar.radio("", [_("Dark Mode"), _("Light Mode")], index=0, key="theme_selection")

        # ユーザーの選択をセッション状態に保存
        st.session_state["theme"] = theme

        # テーマに応じたCSSの適用
        if st.session_state["theme"] == _("Light Mode"):
            st.markdown(light_css, unsafe_allow_html=True)
        else:
            st.markdown(translations.get(LANGUAGES.get(language, "ja"), custom_css), unsafe_allow_html=True)
        
        # ダッシュボードへのアクセスリンク
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + _("Dashboard"))
        if authentication_status:
            if st.button(_("Go to Dashboard")):
                st.session_state["navigate_dashboard"] = True

    elif authentication_status == False:
        st.sidebar.error(_("Incorrect username or password."))
        notify(_("Incorrect username or password."), type="error")
    elif authentication_status == None:
        st.sidebar.warning(_("Please log in."))
        notify(_("Please log in."), type="info")
    
    # ファイル入力
    expander = st.sidebar.expander("🗀 " + _("File Upload"))
    with expander:
        st.text(_("Supports large text/image/URL analysis"))
        image_url = st.text_input(_("Image URL for EXIF analysis:"), key="image_url", on_change=clear_files, value=st.session_state["image_url"])
        file_uploader_key = "file_uploader_{}".format(st.session_state.get("file_uploader_key", False))
        uploaded_files = st.file_uploader(
            _("Upload Files:"),
            type=["txt","pdf","docx","csv","jpg","png","jpeg"],
            key=file_uploader_key,
            on_change=clear_url,
            accept_multiple_files=True,
        )
        if uploaded_files is not None:
            st.session_state["uploaded_files"] = uploaded_files

    # モデル設定
    expander = st.sidebar.expander("⚒ " + _("Model Settings"))
    with expander:
        if "REPLICATE_API_TOKEN" in st.secrets:
            replicate_api = st.secrets["REPLICATE_API_TOKEN"]
        else:
            replicate_api = st.text_input(_("Enter your Replicate API Token:"), type="password")
            if not (replicate_api.startswith("r8_") and len(replicate_api) == 40):
                st.warning(_("Please enter a valid Replicate API token."), icon="⚠️")
                st.markdown(
                    f"**{_('If you do not have an API token, please obtain one from [Replicate](https://replicate.com/account/api-tokens).')}**"
                )
        os.environ["REPLICATE_API_TOKEN"] = replicate_api
        st.subheader(_("Adjust Model Parameters"))
        temperature = st.slider(
            _("Temperature"), min_value=0.01, max_value=5.0, value=0.3, step=0.01
        )
        top_p = st.slider(_("Top P"), min_value=0.01, max_value=1.0, value=0.2, step=0.01)
        max_new_tokens = st.number_input(
            _("Max New Tokens"), min_value=1, max_value=1024, value=512
        )
        min_new_tokens = st.number_input(
            _("Min New Tokens"), min_value=0, max_value=512, value=0
        )
        presence_penalty = st.slider(
            _("Presence Penalty"), min_value=0.0, max_value=2.0, value=1.15, step=0.05
        )
        frequency_penalty = st.slider(
            _("Frequency Penalty"), min_value=0.0, max_value=2.0, value=0.2, step=0.05
        )
        stop_sequences = st.text_area(_("Stop Sequences"), value="<|im_end|>", height=100)

    # 著作権情報の表示
    st.sidebar.caption(
        "All code contributed to Exifa.net is © 2024 by Sahir Maharaj.\n"
        "The content is licensed under the Creative Commons Attribution 4.0 International License.\n"
        "This allows for sharing and adaptation, provided appropriate credit is given, and any changes made are indicated.\n\n"
        f"{_('When using the code from Exifa.net, please credit as follows:')} 'Code sourced from Exifa.net, authored by Sahir Maharaj, 2024.'\n\n"
        f"{_('For reporting bugs, requesting features, or further inquiries, please reach out to Sahir Maharaj at sahir@sahirmaharaj.com.')}\n\n"
        f"{_('Connect with Sahir Maharaj on LinkedIn for updates and potential collaborations')}: https://www.linkedin.com/in/sahir-maharaj/\n\n"
        f"{_('Hire Sahir Maharaj')}: https://topmate.io/sahirmaharaj/362667"
    )

# -----------------------------------
# 5. ファイル処理＆EXIF解析
# -----------------------------------
file_text = ""
if st.session_state["uploaded_files"]:
    for uf in st.session_state["uploaded_files"]:
        if uf.type == "application/pdf":
            with pdfplumber.open(uf) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
            file_text = "\n".join(p for p in pages if p)
        elif uf.type == "text/plain":
            file_text = str(uf.read(), "utf-8")
        elif uf.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uf)
            file_text = "\n".join([para.text for para in doc.paragraphs])
        elif uf.type == "text/csv":
            df = pd.read_csv(uf)
            file_text = df.to_string(index=False)
        elif uf.type in ["image/jpeg","image/png","image/jpg"]:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uf.read())
                temp.flush()
                temp.close()
                with open(temp.name,"rb") as f:
                    tags = exifread.process_file(f)
                os.unlink(temp.name)
            exif_data = {}
            for tag in tags.keys():
                if tag not in ["JPEGThumbnail","TIFFThumbnail","Filename","EXIF MakerNote"]:
                    exif_data[tag] = str(tags[tag])
            df = pd.DataFrame(exif_data, index=[0])
            df.insert(loc=0, column=_("Image Feature"), value=["Value"]*len(df))
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            st.session_state["exif_df"] = df
            file_text = file_text or "\n".join([f"{tag}: {tags[tag]}" for tag in tags.keys() if tag not in ("JPEGThumbnail","TIFFThumbnail","Filename","EXIF MakerNote")])

if st.session_state["image_url"]:
    try:
        resp_head = requests.head(st.session_state["image_url"])
        if resp_head.headers.get("Content-Type","").startswith("image"):
            resp = requests.get(st.session_state["image_url"])
            resp.raise_for_status()
            image_data = BytesIO(resp.content)
            image = Image.open(image_data)
            image.load()
            tags = exifread.process_file(image_data)
            exif_data = {}
            for tag in tags.keys():
                if tag not in ["JPEGThumbnail","TIFFThumbnail","Filename","EXIF MakerNote"]:
                    exif_data[tag] = str(tags[tag])
            df = pd.DataFrame(exif_data, index=[0])
            df.insert(loc=0, column=_("Image Feature"), value=["Value"]*len(df))
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            st.session_state["url_exif_df"] = df
        else:
            st.warning(_("The URL does not point to an image."))
    except:
        st.warning(_("Failed to retrieve image from the URL."))

# -----------------------------------
# 6. メインUI構築
# -----------------------------------
st.markdown("<h1 style='text-align:center;color:white;'>究極融合: EXIF & TTS & GPT & AI分類</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#cccccc;'>300ページ超テキスト音声化、EXIF解析、カラー可視化、GPT対話、AI画像分類</p>", unsafe_allow_html=True)

tabs = st.tabs([_("📜 Text-to-Speech"), _("🖼 EXIF Analysis & Visualization"), _("📊 Dashboard"), _("💬 GPT Conversation"), _("🌐 Profile"), _("📝 Feedback"), _("👤 User Dashboard")])

# テキスト音声合成タブ
with tabs[0]:
    st.subheader(_("Massive Text-to-Speech"))
    input_option = st.selectbox(_("Input Method"), (_("Direct Input"), _("Use Uploaded Text")))
    tts_text = ""
    if input_option == _("Direct Input"):
        tts_text = st.text_area(_("Paste the text to synthesize into speech"), _("Paste large amounts of text here (e.g., full book)"))
    else:
        if file_text:
            st.write(_("Extracted Text (Partial):"))
            st.write(file_text[:500] + _("..."))
            tts_text = file_text
        else:
            st.write(_("No uploaded text available."))
    
    selected_gender = st.selectbox(_("Speaker Gender"), (_("default"), _("male"), _("female"), _("neutral")))
    if tts_text and st.button(_("Execute Speech Synthesis")):
        with st.spinner(_("Synthesizing speech... This may take some time for large texts.")):
            lang_code = detect_language(tts_text)
            final_mp3 = synthesize_speech_chunk(tts_text, lang_code, gender=selected_gender)
        st.success(_("Speech synthesis completed!"))
        notify(_("Speech synthesis was successfully completed."), type="success")
        st.download_button(_("Download MP3"), data=final_mp3, file_name="converted_book.mp3", mime="audio/mpeg")
        st.audio(final_mp3, format="audio/mp3")
        # アクティビティログの記録
        if authentication_status:
            user_obj = db.query(database.User).filter_by(username=username).first()
            if user_obj:
                database.add_activity_log(user_obj.id, _("Executed speech synthesis."), db)

# EXIF解析＆ビジュアルタブ
with tabs[1]:
    st.subheader(_("EXIF Analysis & Visualization"))
    if st.session_state["exif_df"].empty and not st.session_state["image_url"]:
        st.info(_("No EXIF data available: Please upload an image or provide a URL."))
    else:
        st.markdown("##### " + _("EXIF Data Extraction Results"))
        if not st.session_state["exif_df"].empty:
            st.dataframe(st.session_state["exif_df"])
        if not st.session_state["url_exif_df"].empty:
            st.dataframe(st.session_state["url_exif_df"])
        
        image_to_analyze = None
        if st.session_state["uploaded_files"]:
            for f in st.session_state["uploaded_files"]:
                if f.type in ["image/jpeg","image/png","image/jpg"]:
                    image_to_analyze = load_image_cached(f)
                    break
        elif st.session_state["image_url"]:
            image_to_analyze = load_image_cached(st.session_state["image_url"])

        if image_to_analyze:
            st.image(image_to_analyze, caption=_("Uploaded Image"), use_column_width=True)
            data = np.array(image_to_analyze)

            # RGBチャンネル操作
            with st.expander(_("⛆ RGB Channel Manipulation")):
                channels = st.multiselect(_("Select Channels to Display:"), [_("Red"), _("Green"), _("Blue")], default=[_("Red"), _("Green"), _("Blue")])
                if channels:
                    cmap = {_("Red"): 0, _("Green"): 1, _("Blue"): 2}
                    selected_idx = [cmap[ch] for ch in channels]
                    ch_data = np.zeros_like(data)
                    for idx in selected_idx:
                        ch_data[:,:,idx] = data[:,:,idx]
                    st.image(Image.fromarray(ch_data), use_column_width=True)
                else:
                    st.image(image_to_analyze, use_column_width=True)

            # HSVヒストグラム
            with st.expander(_("〽 HSV Histogram")):
                hsv_image = image_to_analyze.convert("HSV")
                hsv_data = np.array(hsv_image)
                hue_hist, _ = np.histogram(hsv_data[:,:,0], bins=256, range=(0,256))
                sat_hist, _ = np.histogram(hsv_data[:,:,1], bins=256, range=(0,256))
                val_hist, _ = np.histogram(hsv_data[:,:,2], bins=256, range=(0,256))
                hsv_histogram_df = pd.DataFrame({"Hue": hue_hist, "Saturation": sat_hist, "Value": val_hist})
                st.line_chart(hsv_histogram_df)

            # カラー分布サンバースト
            with st.expander(_("☄ Color Distribution Sunburst")):
                red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
                ci = {"color": [], "intensity": [], "count": []}
                for name, channel in zip([_("Red"), _("Green"), _("Blue")], [red, green, blue]):
                    unique, counts = np.unique(channel, return_counts=True)
                    ci["color"].extend([name]*len(unique))
                    ci["intensity"].extend(unique)
                    ci["count"].extend(counts)
                cdf = pd.DataFrame(ci)
                fig = px.sunburst(
                    cdf,
                    path=["color", "intensity"],
                    values="count",
                    color="color",
                    color_discrete_map={_("Red"): "#ff6666", _("Green"): "#85e085", _("Blue"): "#6666ff"}
                )
                st.plotly_chart(fig, use_container_width=True)

            # 3D色空間プロット
            with st.expander(_("🕸 3D Color Space Plot")):
                skip = 8
                sample = data[::skip, ::skip].reshape(-1, 3)
                fig = go.Figure(data=[go.Scatter3d(
                    x=sample[:,0], y=sample[:,1], z=sample[:,2],
                    mode="markers",
                    marker=dict(size=3, color=["rgb({},{},{})".format(r,g,b) for r,g,b in sample])
                )])
                fig.update_layout(scene=dict(xaxis_title=_("Red"), yaxis_title=_("Green"), zaxis_title=_("Blue")))
                st.plotly_chart(fig, use_container_width=True)

            # 画像分類機能の追加
            with st.expander(_("🤖 Image Classification")):
                st.markdown("### " + _("AI-based Image Classification"))
                if replicate_api and st.session_state["username"]:
                    classify_button = st.button(_("Classify Image"))
                    if classify_button:
                        with st.spinner(_("Classifying image...")):
                            # 画像を一時ファイルとして保存
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                                image_to_analyze.save(temp.name)
                                temp_path = temp.name
                            
                            # 画像分類を実行
                            prediction = classify_image(temp_path, replicate_api)
                            
                            # 一時ファイルを削除
                            os.unlink(temp_path)
                            
                            if prediction:
                                # 分類結果をデータベースに保存
                                user_obj = db.query(database.User).filter_by(username=username).first()
                                if user_obj:
                                    database.add_image_classification(
                                        user_id=user_obj.id,
                                        image_path=temp_path,
                                        classification_result=json.dumps(prediction, ensure_ascii=False)
                                    )
                                    database.add_activity_log(user_obj.id, _("Executed image classification."), db)
                                
                                # 分類結果を表示
                                st.markdown("#### " + _("Classification Results"))
                                st.json(prediction)
                                notify(_("Image classification completed successfully."), type="success")
                            else:
                                notify(_("Image classification failed."), type="error")
                else:
                    st.warning(_("Replicate API token is not set or user is not logged in."))

            # EXIF除去後の画像ダウンロード
            st.markdown("#### " + _("Download Image without EXIF Data"))
            cleaned = clear_exif_data(image_to_analyze)
            download_image(cleaned)
            notify(_("Image with EXIF data removed is ready for download."), type="info")
            # アクティビティログの記録
            if authentication_status:
                user_obj = db.query(database.User).filter_by(username=username).first()
                if user_obj:
                    database.add_activity_log(user_obj.id, _("Downloaded image without EXIF data."), db)

        # 簡易コメント（LLM対応可）
        if not st.session_state["exif_df"].empty:
            commentary = _("EXIFから撮影者の機材・露出設定などが推測可能。撮影環境は自然光か計画的照明下とみられ、撮影者は中級的経験と程よい予算を持つと考えられる。")
            st.markdown("#### " + _("Auto-generated Commentary"))
            st.write(commentary)
            if st.button(_("Play Commentary Audio")):
                lang_code = detect_language(commentary)
                audio_data = synthesize_speech_chunk(commentary, lang_code)
                st.audio(audio_data, format="audio/mp3")
                notify(_("Commentary audio playback completed."), type="success")
                # アクティビティログの記録
                if authentication_status:
                    user_obj = db.query(database.User).filter_by(username=username).first()
                    if user_obj:
                        database.add_activity_log(user_obj.id, _("Played commentary audio."), db)

# ダッシュボードタブ
with tabs[2]:
    st.subheader(_("Dashboard"))
    
    if st.session_state["exif_df"].empty and st.session_state["url_exif_df"].empty:
        st.info(_("No EXIF data available. Please upload an image or provide a URL."))
    else:
        combined_exif_df = pd.DataFrame()
        if not st.session_state["exif_df"].empty:
            combined_exif_df = combined_exif_df.append(st.session_state["exif_df"], ignore_index=True)
        if not st.session_state["url_exif_df"].empty:
            combined_exif_df = combined_exif_df.append(st.session_state["url_exif_df"], ignore_index=True)
        
        st.dataframe(combined_exif_df)
        
        plot_exif_statistics(combined_exif_df)
        notify(_("Dashboard has been updated."), type="info")
        # アクティビティログの記録
        if authentication_status:
            user_obj = db.query(database.User).filter_by(username=username).first()
            if user_obj:
                database.add_activity_log(user_obj.id, _("Viewed dashboard."), db)

# GPT対話タブ
with tabs[3]:
    st.subheader(_("GPT-based Advanced Conversation"))
    
    # アイコンURLの定義（例として）
    icons = {
        "user": "https://img.icons8.com/ios-glyphs/30/000000/user--v1.png",
        "assistant": "https://img.icons8.com/ios-glyphs/30/000000/bot.png"
    }
    
    # チャットメッセージの表示
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=icons.get(msg["role"], "")):
            st.write(msg["content"])
    
    user_input = st.chat_input(_("Feel free to ask anything, including EXIF, TTS, image analysis, color space, AI image classification, or general questions."))
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=icons.get("user", "")):
            st.write(user_input)
        # アクティビティログの記録
        if authentication_status:
            user_obj = db.query(database.User).filter_by(username=username).first()
            if user_obj:
                database.add_activity_log(user_obj.id, _("Executed GPT conversation."), db)
    
        # アシスタントのメッセージを準備
        assistant_placeholder = st.chat_message("assistant", avatar=icons.get("assistant", ""))
        assistant_placeholder.markdown(_("**Responding...**"))
    
        # GPTのストリーミングレスポンスを処理
        response_stream = generate_gpt_response_stream(st.session_state["messages"], db)
        full_response = ""
        try:
            for chunk in response_stream:
                if _("An error has occurred") in chunk:
                    assistant_placeholder.markdown(f"**{_('Error')}:** {chunk}")
                    break
                full_response += chunk
                assistant_placeholder.markdown(full_response + "▌")  # タイピングインジケーター
                time.sleep(0.05)  # ストリーミング速度調整
            # 最終的なメッセージを更新
            st.session_state["messages"].append({"role": "assistant", "content": full_response})
            assistant_placeholder.markdown(full_response)  # タイピングインジケーターを削除
            notify(_("GPT conversation has been updated."), type="success")
            # アクティビティログの記録
            if authentication_status:
                user_obj = db.query(database.User).filter_by(username=username).first()
                if user_obj:
                    database.add_activity_log(user_obj.id, _("GPT conversation updated."), db)
        except Exception as e:
            assistant_placeholder.markdown(f"**{_('Error Occurred')}:** {str(e)}")
            notify(f"{_('An error occurred during GPT conversation')}: {str(e)}", type="error")
            # アクティビティログの記録
            if authentication_status:
                user_obj = db.query(database.User).filter_by(username=username).first()
                if user_obj:
                    database.add_activity_log(user_obj.id, _("Error occurred during GPT conversation."), db)

# プロフィールタブ
with tabs[4]:
    st.subheader(_("🌐 Profile"))
    st.markdown("#### " + _("Masaki Kusaka's Link Collection"))
    
    # プロフィール写真と紹介
    col1, col2 = st.columns([1, 2])
    with col1:
        profile_pic = "https://avatars.githubusercontent.com/u/your_github_id?s=400&u=your_profile_pic_url&v=4"  # ここにプロフィール写真のURLを挿入
        st.image(profile_pic, width=150, caption=_("Masaki Kusaka"))
    with col2:
        st.write("""
            **Masaki Kusaka** is a software developer, data scientist, and AI engineer. Utilizing the latest technologies, he provides innovative solutions. Check out my projects and activities through the links below.
        """)
    
    # リンク集
    st.markdown("<div class='profile-links'>", unsafe_allow_html=True)
    links = {
        "GitHub": "https://github.com/MKYUKI",
        "YouTube": "https://www.youtube.com/@mk_agi",
        "PayPal": "https://www.paypal.com/paypalme/MasakiKusaka",
        "Dropbox": "https://www.dropbox.com/home",
        "HuggingFace": "https://huggingface.co/pricing",
        "X (旧Twitter)": "https://x.com/MK_ASI1",
        "Facebook": "https://www.facebook.com/",
        "Amazon JP": "https://www.amazon.co.jp/s?i=digital-text&rh=p_27%3AMasaki+Kusaka&s=relevancerank&text=Masaki+Kusaka&ref=dp_byline_sr_ebooks_1",
        "Amazon US": "https://www.amazon.com/s?i=digital-text&rh=p_27%3AMasaki+Kusaka&s=relevancerank&text=Masaki+Kusaka&ref=dp_byline_sr_ebooks_1",
        "YouTube Repo": "https://github.com/MKYUKI/youtube-new.git"
    }

    icons = {
        "GitHub": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        "YouTube": "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg",
        "PayPal": "https://www.paypalobjects.com/webstatic/icon/pp258.png",
        "Dropbox": "https://cfl.dropboxstatic.com/static/images/favicon-vfl8kIxhC.ico",
        "HuggingFace": "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        "X (旧Twitter)": "https://abs.twimg.com/icons/apple-touch-icon-192x192.png",
        "Facebook": "https://www.facebook.com/images/fb_icon_325x325.png",
        "Amazon JP": "https://m.media-amazon.com/images/G/01/gc/designs/livepreview/amazonlogo._CB485932169_.png",
        "Amazon US": "https://m.media-amazon.com/images/G/01/gc/designs/livepreview/amazonlogo._CB485932169_.png",
        "YouTube Repo": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
    }

    for name, url in links.items():
        icon_url = icons.get(name, "")
        st.markdown(f"""
            <a href="{url}" target="_blank">
                <img src="{icon_url}" alt="{name}" title="{name}" style="width:30px;height:30px;margin-right:10px;">
            </a>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# フィードバックタブの実装
with tabs[5]:
    if is_admin(username, db):
        admin_feedback_view(db)
        st.markdown("---")
        admin_image_classification_view(db)
    else:
        st.subheader(_("Feedback"))
        with st.form("user_feedback"):
            feedback = st.text_area(_("Please provide your feedback or comments here."), "")
            submit_feedback = st.form_submit_button(_("Send"))
        if submit_feedback and feedback:
            user_obj = db.query(database.User).filter_by(username=username).first()
            if user_obj:
                database.add_feedback(user_obj.id, feedback, db)
                database.add_activity_log(user_obj.id, _("Submitted feedback."), db)
                st.success(_("Thank you for your feedback!"))
                notify(_("Feedback has been successfully submitted."), type="success")
            else:
                st.error(_("Failed to submit feedback. Please log in again."))
                notify(_("Failed to submit feedback. Please log in again."), type="error")
        elif submit_feedback:
            st.warning(_("Please enter your feedback before submitting."))
            notify(_("Please enter your feedback before submitting."), type="info")

# ユーザーダッシュボードタブの実装
with tabs[6]:
    if authentication_status:
        st.subheader(_("👤 User Dashboard"))
        
        user_obj = db.query(database.User).filter_by(username=username).first()
        if user_obj:
            # プロフィール情報の表示・編集
            st.markdown("### " + _("Profile Information"))
            with st.expander(_("🔧 Edit Profile")):
                new_name = st.text_input(_("Name"), value=user_obj.name)
                new_email = st.text_input(_("Email"), value=user_obj.email if user_obj.email else "")
                new_password = st.text_input(_("New Password"), type="password")
                confirm_new_password = st.text_input(_("Confirm New Password"), type="password")
                submitted = st.button(_("Update Profile"))
                if submitted:
                    update_flag = False
                    if new_password:
                        if new_password != confirm_new_password:
                            st.error(_("New passwords do not match."))
                            notify(_("New passwords do not match."), type="error")
                        else:
                            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                            user_obj.password = hashed_password
                            database.add_activity_log(user_obj.id, _("Updated password."), db)
                            notify(_("Password has been updated."), type="success")
                            update_flag = True
                    if new_email and new_email != user_obj.email:
                        if database.email_exists(new_email, db):
                            st.error(_("This email address is already in use."))
                            notify(_("This email address is already in use."), type="error")
                        else:
                            user_obj.email = new_email
                            database.add_activity_log(user_obj.id, _("Updated email address."), db)
                            notify(_("Email address has been updated."), type="success")
                            update_flag = True
                    if new_name and new_name != user_obj.name:
                        user_obj.name = new_name
                        database.add_activity_log(user_obj.id, _("Updated name."), db)
                        notify(_("Name has been updated."), type="success")
                        update_flag = True
                    if update_flag:
                        db.commit()
            
            # 通知設定の表示・編集
            st.markdown("### " + _("Notification Settings"))
            user_settings = database.get_user_settings(user_obj.id, db)
            if user_settings:
                with st.expander(_("🔧 Edit Notification Settings")):
                    notify_tts = st.checkbox(_("Receive notifications when speech synthesis is complete"), value=bool(user_settings.notify_tts))
                    notify_classification = st.checkbox(_("Receive notifications when image classification is complete"), value=bool(user_settings.notify_classification))
                    notify_feedback = st.checkbox(_("Receive notifications when feedback is submitted"), value=bool(user_settings.notify_feedback))
                    submitted_settings = st.button(_("Update Notification Settings"))
                    if submitted_settings:
                        database.update_user_settings(
                            user_id=user_obj.id,
                            notify_tts=int(notify_tts),
                            notify_classification=int(notify_classification),
                            notify_feedback=int(notify_feedback),
                            db=db
                        )
                        database.add_activity_log(user_obj.id, _("Updated notification settings."), db)
                        notify(_("Notification settings have been updated."), type="success")
            
            # アクティビティログの表示
            st.markdown("### " + _("Activity Log"))
            activities = database.get_user_activities(user_obj.id, db)
            if not activities:
                st.info(_("No activity logs available yet."))
            else:
                for act in activities:
                    st.markdown(f"- **{act.timestamp.strftime('%Y-%m-%d %H:%M:%S')}**: {act.activity}")
            
            # データ管理
            st.markdown("### " + _("Data Management"))
            with st.expander(_("🔧 Manage Data")):
                st.markdown("#### " + _("Uploaded Images"))
                user_classifications = db.query(database.ImageClassification).filter_by(user_id=user_obj.id).all()
                if not user_classifications:
                    st.info(_("No uploaded images found."))
                else:
                    for cls in user_classifications:
                        st.image(cls.image_path, width=200, caption=f"{_('Classification Result')}: {cls.classification_result}")
                        # ダウンロードボタン
                        with open(cls.image_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        st.download_button(
                            label=_("Download Image"),
                            data=img_bytes,
                            file_name=os.path.basename(cls.image_path),
                            mime="image/jpeg",
                        )
                        st.markdown("---")
                
                st.markdown("#### " + _("Generated Audio Files"))
                # ここでは、音声ファイルの管理機能を実装します。音声ファイルを保存し、そのパスをデータベースに記録する必要があります。
                # 既存の音声合成機能では音声ファイルを一時的に作成していますが、永続的に保存する機能を追加することを推奨します。
                st.info(_("Audio file management functionality is not yet implemented."))
        
        else:
            st.error(_("Failed to retrieve user information."))
    else:
        st.warning(_("Please log in."))

# フッター
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1e1e1e;
    color: white;
    text-align: center;
    padding: 10px 0;
}
.footer a {
    color: #ffffff;
    text-decoration: none;
    margin: 0 10px;
}
.footer a:hover {
    text-decoration: underline;
}
.profile-links img {
    width: 35px;
    height: 35px;
}
</style>
<div class="footer">
    © Exifa.net (Sahir Maharaj,2024), CC-BY 4.0. {(_('An all-in-one ultimate app integrated into the world for the first time.'))}<br>
    <a href="mailto:sahir@sahirmaharaj.com">{_('Contact Us')}</a> |
    <a href="https://www.linkedin.com/in/sahir-maharaj/" target="_blank">{_('LinkedIn')}</a> |
    <a href="https://topmate.io/sahirmaharaj/362667" target="_blank">{_('Hire me!')}</a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
