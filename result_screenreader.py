import streamlit as st
from streamlit.components.v1 import html
from cryptography.fernet import Fernet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from io import BytesIO
import os
import hashlib
import base64
import json
import re
from datetime import datetime
from gtts import gTTS
from difflib import SequenceMatcher
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
from aiortc.contrib.media import MediaRecorder
import soundfile as sf
from pathlib import Path
import time
import pydub
import whisper
import av
import numpy as np
from openai import OpenAI

# QR 코드 생성을 위한 라이브러리
try:
    import qrcode
    from PIL import Image
    from reportlab.platypus import Image as RLImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# AWS S3 업로드를 위한 라이브러리
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from boto3.exceptions import S3UploadFailedError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 한글 폰트 등록
def register_korean_fonts():
    """한글 폰트를 등록합니다."""
    try:
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",   # 바탕
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                if font_path.endswith('.ttf'):
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    return 'Korean'
                elif font_path.endswith('.ttc'):
                    pdfmetrics.registerFont(TTFont('Korean', font_path, subfontIndex=0))
                    return 'Korean'
        
        return 'Helvetica'
    except Exception as e:
        return 'Helvetica'

KOREAN_FONT = register_korean_fonts()

# 오디오 녹음 파일 저장 경로
TMP_DIR = Path("C:/audio/sound")
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

if "wavpath" not in st.session_state:
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    st.session_state["wavpath"] = str(TMP_DIR / f"{cur_time}.wav")

wavpath = st.session_state["wavpath"]

# 오디오 입력 설정
MEDIA_STREAM_CONSTRAINTS = {
    "video": False,
    "audio": {
        "echoCancellation": False,
        "noiseSuppression": True,
        "autoGainControl": True,
    },
}

# 오디오 프레임 버퍼 클래스
class AudioFrameBuffer:
    def __init__(self):
        self._audio_segments = []  # pydub AudioSegment 리스트로 직접 저장

    def append(self, frame: av.AudioFrame):
        """오디오 프레임을 직접 pydub AudioSegment로 변환하여 저장 (원본 그대로 유지)"""
        # WebRTC 오디오 프레임을 직접 pydub AudioSegment로 변환
        # 이 방식이 원본 샘플 레이트와 속도를 정확히 유지합니다
        sound = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,  # 원본 샘플 레이트 사용
            channels=len(frame.layout.channels),
        )
        self._audio_segments.append(sound)

    def clear(self):
        self._audio_segments.clear()

    def to_pydub_audiosegment(self):
        """모든 오디오 세그먼트를 합쳐서 하나의 AudioSegment로 반환"""
        if not self._audio_segments:
            return pydub.AudioSegment.empty()
        
        # 모든 세그먼트를 연결 (원본 속도와 샘플 레이트 유지)
        result = self._audio_segments[0]
        for segment in self._audio_segments[1:]:
            result += segment
        return result

    def to_wav_file(self, wavpath):
        """WAV 파일로 저장 - 원본 샘플 레이트와 속도 유지"""
        if not self._audio_segments:
            return False
        
        audio_segment = self.to_pydub_audiosegment()
        if len(audio_segment) > 0:
            # 원본 그대로 저장 (피치나 속도 변경 없음)
            audio_segment.export(wavpath, format="wav")
            return True
        return False

# 오디오 프로세서 클래스
class AudioProcessor:
    def __init__(self, buffer: AudioFrameBuffer):
        self.buffer = buffer

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.buffer.append(frame)
        return frame

# 오디오 프레임 수집 -> pydub으로 저장
def save_frames_from_audio_receiver(wavpath):
    # 세션 상태 초기화
    if "audio_buffer_obj" not in st.session_state:
        st.session_state["audio_buffer_obj"] = AudioFrameBuffer()

    buffer = st.session_state["audio_buffer_obj"]
    
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        audio_processor_factory=lambda: AudioProcessor(buffer),
    )

    # 녹음이 끝나면 버퍼를 WAV로 저장
    if webrtc_ctx.state.playing is False and len(buffer._audio_segments) > 0:
        if buffer.to_wav_file(wavpath):
            buffer.clear()
            st.success("녹음이 완료되었습니다.")
            # 스크린리더 안내
            if st.session_state.screen_reader_enabled:
                screen_reader_announce_sync("녹음이 완료되었습니다. Whisper로 텍스트 변환 버튼을 눌러주세요.")

# 저장된 wav 파일 재생
def display_wavfile(wavpath):
    with open(wavpath, 'rb') as f:
        audio_bytes = f.read()
    file_type = Path(wavpath).suffix
    st.audio(audio_bytes, format=f'audio/{file_type}', start_time=0)

# ==========================================
# [공용 함수] 텍스트 → 오디오 재생 함수
# ==========================================
def tts_play(text):
    """문자를 음성(mp3)으로 생성 후 HTML로 재생"""
    try:
        tts = gTTS(text=text, lang='ko')
        mp3 = BytesIO()
        tts.write_to_fp(mp3)
        mp3.seek(0)
        b64 = base64.b64encode(mp3.read()).decode()

        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"오디오 재생 오류: {e}")

def screen_reader_announce(text, priority="polite"):
    """스크린리더를 위한 자동 음성 안내
    
    Args:
        text: 읽을 텍스트
        priority: "polite" (기본) 또는 "assertive" (긴급)
    """
    if st.session_state.get("screen_reader_enabled", False):
        try:
            # Web Speech API를 사용한 브라우저 내장 TTS (더 빠름)
            announcement_html = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    const utterance = new SpeechSynthesisUtterance('{text}');
                    utterance.lang = 'ko-KR';
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    speechSynthesis.speak(utterance);
                }} else {{
                    // Web Speech API가 없으면 fallback으로 TTS 사용
                    console.log('Web Speech API not supported');
                }}
            </script>
            """
            st.markdown(announcement_html, unsafe_allow_html=True)
        except Exception:
            # Web Speech API 실패 시 기존 TTS 사용
            tts_play(text)

def screen_reader_announce_sync(text):
    """스크린리더 동기식 안내 (기존 TTS 사용, 더 안정적)"""
    if st.session_state.get("screen_reader_enabled", False):
        tts_play(text)

# ==========================================
# [gpt.py에서 가져온 함수들]
# ==========================================

def extract_personal_info(text):
    """텍스트에서 개인정보를 추출합니다."""
    prompt = f"""
    다음 텍스트에서 개인정보를 추출해 JSON으로 정리해줘.

    반드시 아래 key만 사용해서 JSON으로 출력해.
    없는 값은 "" (빈 문자열) 로 넣어.

    keys:
    - name
    - rrn
    - address
    - phone
    - birthdate
    - employer

    텍스트:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 개인정보 정보를 정리하고, 반드시 JSON 형식으로만 응답해야 합니다."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result_text = response.choices[0].message.content.strip()
    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError as e:
        st.error(f"JSON 파싱 실패: {e}")
        raise

    return result_json

def generate_document_content(info_json, doc_type="근로계약서"):
    """개인정보를 바탕으로 문서 내용을 생성합니다."""
    info_str = json.dumps(info_json, indent=2, ensure_ascii=False)
    
    doc_type_prompts = {
        "개인정보 제공 동의서": "당신은 개인정보 제공 동의서의 본문 내용을 작성하는 전문가입니다. 제공된 개인 정보를 바탕으로 동의서에 들어갈 본문 내용만 작성하세요. 동의 목적, 항목, 기간 등을 설명하는 본문 내용을 공식적인 용어로 작성하세요. 형식이나 구조는 작성하지 말고, 본문 내용에만 집중하세요.",
        "주민등록등본 발급 신청서": "당신은 주민등록등본 발급 신청서의 신청 사유 및 내용을 작성하는 전문가입니다. 제공된 개인 정보를 바탕으로 신청서에 들어갈 신청 사유와 내용만 작성하세요. 신청 사유 및 목적을 법적 근거를 바탕으로 공식적인 문체로 작성하세요. 형식이나 구조는 작성하지 말고, 본문 내용에만 집중하세요.",
        "주민등록등본 신청서": "당신은 주민등록등본 발급 신청서의 신청 사유 및 내용을 작성하는 전문가입니다. 제공된 개인 정보를 바탕으로 신청서에 들어갈 신청 사유와 내용만 작성하세요. 신청 사유 및 목적을 법적 근거를 바탕으로 공식적인 문체로 작성하세요. 형식이나 구조는 작성하지 말고, 본문 내용에만 집중하세요.",
        "근로계약서": "당신은 근로계약서의 근로 조건 및 내용을 작성하는 전문가입니다. 제공된 개인 정보를 바탕으로 근로계약서에 들어갈 근로 조건, 직무 내용, 급여 등 본문 내용만 작성하세요. 표준 근로계약서의 핵심 조항(직무, 급여, 근무 시간)에 대한 내용을 법률 용어와 객관적 사실만을 사용하여 작성하세요. 형식이나 구조는 작성하지 말고, 본문 내용에만 집중하세요."
    }
    
    system_prompt = doc_type_prompts.get(doc_type, f"당신은 {doc_type}의 본문 내용을 작성하는 전문가입니다. 제공된 개인 정보를 활용하여 문서에 들어갈 본문 내용만 작성하세요. 형식이나 구조는 작성하지 말고, 본문 내용에만 집중하세요.")

    prompt = f"""
다음 개인 정보를 활용하여 "{doc_type}"에 들어갈 본문 내용을 작성해주세요.

**작성 지침:**
1. 제공된 개인 정보를 정확하게 반영하세요.
2. 문서의 형식이나 구조는 작성하지 말고, 본문 내용만 작성하세요.
3. 자연스럽고 읽기 쉬운 문장으로 작성하세요.
4. 개인 정보가 없는 항목은 적절히 처리하거나 생략하세요.
5. 문서 유형에 맞는 적절한 톤과 스타일을 유지하세요.

**개인 정보:**
{info_str}

위 정보를 바탕으로 {doc_type}의 본문 내용만 작성해주세요.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        document_content = response.choices[0].message.content.strip()
        return document_content
    
    except Exception as e:
        st.error(f"문서 생성 중 오류 발생: {e}")
        raise

def calculate_document_hash(filepath):
    """PDF 파일의 해시값을 계산합니다."""
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        return None

def generate_qr_code(data, output_file='qrcode.png', size=200):
    """QR 코드를 생성합니다."""
    if not QR_AVAILABLE:
        return None
    
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(output_file)
        
        return output_file
    except Exception as e:
        return None

def upload_audio_to_s3(audio_filepath, bucket_name=None, s3_key=None, region='ap-northeast-2'):
    """음성 파일을 AWS S3에 업로드하고 공개 URL을 반환합니다."""
    if not S3_AVAILABLE:
        return None
    
    if not os.path.exists(audio_filepath):
        return None
    
    if not bucket_name:
        bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if region == 'ap-northeast-2':
        region = os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or region
    
    if not bucket_name:
        return None
    
    if not s3_key:
        date_folder = datetime.now().strftime("%Y/%m/%d")
        filename = os.path.basename(audio_filepath)
        s3_key = f"audio/{date_folder}/{filename}"
    
    try:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if aws_access_key_id and aws_secret_access_key:
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            s3_client = boto3.client('s3', region_name=region)
        
        try:
            s3_client.upload_file(
                audio_filepath,
                bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'audio/wav',
                    'ACL': 'public-read'
                }
            )
        except (ClientError, S3UploadFailedError) as acl_error:
            error_str = str(acl_error)
            if 'AccessControlListNotSupported' in error_str or 'InvalidRequest' in error_str:
                s3_client.upload_file(
                    audio_filepath,
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'audio/wav'
                    }
                )
            else:
                raise
        
        public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        return public_url
        
    except Exception as e:
        return None

def upload_audio_to_web_server(audio_filepath, base_url=None):
    """음성 파일을 웹 서버에 업로드하고 공개 URL을 반환합니다."""
    s3_url = upload_audio_to_s3(audio_filepath)
    if s3_url:
        return s3_url
    
    if not os.path.exists(audio_filepath):
        return None
    
    if not base_url:
        base_url = os.getenv("WEB_SERVER_URL", "https://example.com/audio")
    
    filename = os.path.basename(audio_filepath)
    public_url = f"{base_url.rstrip('/')}/{filename}"
    
    return public_url

def get_audio_file_url(audio_filepath, use_web_url=True):
    """음성 파일의 접근 가능한 URL을 생성합니다."""
    if use_web_url:
        web_url = upload_audio_to_web_server(audio_filepath)
        if web_url:
            return web_url
    
    if os.path.exists(audio_filepath):
        return os.path.abspath(audio_filepath)
    return audio_filepath

def calculate_text_similarity(text1, text2):
    """두 텍스트의 유사도를 계산합니다 (0.0 ~ 1.0)."""
    # 공백 제거 및 소문자 변환으로 정규화
    text1_normalized = re.sub(r'\s+', '', text1.lower())
    text2_normalized = re.sub(r'\s+', '', text2.lower())
    
    # SequenceMatcher를 사용한 유사도 계산
    similarity = SequenceMatcher(None, text1_normalized, text2_normalized).ratio()
    return similarity

def verify_consent_phrase(audio_filepath, target_phrase="본인은 상기 내용을 확인하고 이에 동의합니다.", threshold=0.6):
    """음성 파일을 텍스트로 변환하고 동의 문구와의 유사도를 검증합니다.
    
    Args:
        audio_filepath: 검증할 오디오 파일 경로
        target_phrase: 목표 동의 문구
        threshold: 최소 유사도 임계값 (기본값: 0.6 = 60%)
    
    Returns:
        tuple: (유사도, 변환된 텍스트, 검증 통과 여부)
    """
    if not os.path.exists(audio_filepath):
        return None, None, False
    
    try:
        # Whisper로 음성을 텍스트로 변환
        if "whisper_model" not in st.session_state:
            st.session_state.whisper_model = whisper.load_model("small")
        model = st.session_state.whisper_model
        result = model.transcribe(str(audio_filepath))
        transcribed_text = result["text"].strip()
        
        # 유사도 계산
        similarity = calculate_text_similarity(transcribed_text, target_phrase)
        
        # 임계값 이상이면 통과
        is_valid = similarity >= threshold
        
        return similarity, transcribed_text, is_valid
    except Exception as e:
        st.error(f"음성 검증 중 오류: {str(e)}")
        return None, None, False

def create_voice_signature(document_content, pdf_filepath, audio_filepath='recorded_audio.wav'):
    """음성 서명 데이터를 생성합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    document_hash = calculate_document_hash(pdf_filepath) if os.path.exists(pdf_filepath) else None
    
    audio_file_size = os.path.getsize(audio_filepath) if os.path.exists(audio_filepath) else 0
    audio_file_url = os.path.abspath(audio_filepath) if os.path.exists(audio_filepath) else None
    
    voice_signature = {
        "timestamp": timestamp,
        "document_hash": document_hash,
        "audio_file_path": audio_file_url,
        "audio_file_size": audio_file_size,
        "consent_phrase": "본인은 상기 내용을 확인하고 이에 동의합니다."
    }
    
    return voice_signature

def save_voice_signature(voice_signature, output_dir="documents"):
    """음성 서명 데이터를 JSON 파일로 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    signature_file = os.path.join(output_dir, f"voice_signature_{timestamp_str}.json")
    
    with open(signature_file, 'w', encoding='utf-8') as f:
        json.dump(voice_signature, f, indent=2, ensure_ascii=False)
    
    return signature_file

def generate_document(info_json, doc_type="근로계약서", save_file=True, output_dir="documents"):
    """추출된 JSON 정보와 문서 유형을 바탕으로 문서를 생성하고 파일로 저장합니다."""
    document_content = generate_document_content(info_json, doc_type)
    
    if save_file:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        name = info_json.get("name", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{doc_type}_{timestamp}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            create_document_pdf(document_content, doc_type, info_json, f, voice_signature=None)
        
        return document_content, filepath
    
    return document_content, None

def get_pdf_styles():
    """모든 PDF 스타일을 중앙에서 정의하고 반환합니다."""
    styles = getSampleStyleSheet()
    
    pdf_styles = {
        'DocTitle': ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName=KOREAN_FONT,
            fontSize=16,
            textColor='#000000',
            spaceAfter=15,
            alignment=TA_CENTER
        ),
        'TableLabelStyle': ParagraphStyle(
            'TableLabelStyle',
            parent=styles['Normal'],
            fontName=KOREAN_FONT,
            fontSize=10,
            textColor='#000000',
            alignment=TA_LEFT
        ),
        'TableValueStyle': ParagraphStyle(
            'TableValueStyle',
            parent=styles['Normal'],
            fontName=KOREAN_FONT,
            fontSize=10,
            textColor='#000000',
            alignment=TA_LEFT
        ),
        'ContentStyle': ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontName=KOREAN_FONT,
            fontSize=10,
            leading=14,
            textColor='#000000',
            alignment=TA_LEFT
        ),
        'GenericTitle': ParagraphStyle(
            'GenericTitle',
            parent=styles['Heading1'],
            fontName=KOREAN_FONT,
            fontSize=18,
            textColor='#000000',
            spaceAfter=12,
            alignment=TA_CENTER
        ),
        'GenericBody': ParagraphStyle(
            'GenericBody',
            parent=styles['Normal'],
            fontName=KOREAN_FONT,
            fontSize=11,
            leading=18,
            textColor='#000000',
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
    }
    
    return pdf_styles

PDF_STYLES = get_pdf_styles()

def create_paragraph(text, style_name):
    """Paragraph 객체를 생성하는 헬퍼 함수."""
    if not text:
        text = ""
    
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*]+)\*', r'<b>\1</b>', text)
    
    tag_placeholders = {}
    protected_text = text
    tag_counter = 0
    
    def replace_tag(match):
        nonlocal tag_counter
        tag = match.group(0)
        placeholder = f'__HTML_TAG_{tag_counter}__'
        tag_placeholders[placeholder] = tag
        tag_counter += 1
        return placeholder
    
    protected_text = re.sub(r'<[^>]+>', replace_tag, protected_text)
    escaped_text = protected_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    for placeholder, tag in tag_placeholders.items():
        escaped_text = escaped_text.replace(placeholder, tag)
    
    escaped_text = escaped_text.replace('\n', '<br/>')
    
    style = PDF_STYLES.get(style_name, PDF_STYLES['GenericBody'])
    return Paragraph(escaped_text, style)

def create_application_form_pdf(content, doc_type, info_json, buffer, voice_signature=None):
    """신청서 형식의 구조화된 PDF를 생성합니다."""
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25*mm,
        leftMargin=25*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    story = []
    
    # 제목
    story.append(create_paragraph(f"<b>{doc_type}</b>", 'DocTitle'))
    story.append(Spacer(1, 10*mm))
    
    # 개인정보 테이블
    data = [
        [create_paragraph("<b>항목</b>", 'TableLabelStyle'), 
         create_paragraph("<b>내용</b>", 'TableLabelStyle')],
        [create_paragraph("성명", 'TableLabelStyle'), 
         create_paragraph(info_json.get("name", ""), 'TableValueStyle')],
        [create_paragraph("생년월일", 'TableLabelStyle'), 
         create_paragraph(info_json.get("birthdate", ""), 'TableValueStyle')],
        [create_paragraph("주민등록번호", 'TableLabelStyle'), 
         create_paragraph(info_json.get("rrn", ""), 'TableValueStyle')],
        [create_paragraph("주소", 'TableLabelStyle'), 
         create_paragraph(info_json.get("address", ""), 'TableValueStyle')],
        [create_paragraph("연락처", 'TableLabelStyle'), 
         create_paragraph(info_json.get("phone", ""), 'TableValueStyle')],
    ]
    
    if info_json.get("employer") and doc_type != "주민등록등본 발급 신청서" and doc_type != "주민등록등본 신청서":
        data.append([
            create_paragraph("회사명", 'TableLabelStyle'), 
            create_paragraph(info_json.get("employer", ""), 'TableValueStyle')
        ])
    
    table = Table(data, colWidths=[40*mm, 120*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), KOREAN_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 10*mm))
    
    # 신청 사유/내용 섹션
    story.append(create_paragraph("<b>■ 신청 사유 및 내용</b>", 'TableLabelStyle'))
    story.append(Spacer(1, 5*mm))
    
    content_data = [
        [create_paragraph(content, 'ContentStyle')]
    ]
    content_table = Table(content_data, colWidths=[160*mm])
    content_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(content_table)
    story.append(Spacer(1, 15*mm))
    
    # 전자 서명 메타데이터 서명란
    if voice_signature:
        story.append(create_paragraph("<b>■ 전자 서명 및 증거 메타데이터</b>", 'TableLabelStyle'))
        story.append(Spacer(1, 5*mm))
        
        metadata_rows = []
        signer_name = info_json.get("name", "미상")
        metadata_rows.append([
            create_paragraph("전자 서명 주체", 'TableLabelStyle'),
            create_paragraph(f"신청인: {signer_name} (음성 동의 완료)", 'TableValueStyle')
        ])
        
        timestamp = voice_signature.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        metadata_rows.append([
            create_paragraph("전자 서명 일시", 'TableLabelStyle'),
            create_paragraph(timestamp, 'TableValueStyle')
        ])
        
        doc_hash = voice_signature.get("document_hash", "")
        if doc_hash:
            hash_display = f"{doc_hash[:16]}...{doc_hash[-8:]}"
            metadata_rows.append([
                create_paragraph("문서 해시", 'TableLabelStyle'),
                create_paragraph(f"SHA-256: {hash_display}", 'TableValueStyle')
            ])
        
        # QR 코드 생성 및 삽입
        audio_url = get_audio_file_url(voice_signature.get("audio_file_path", ""), use_web_url=True)
        if audio_url and QR_AVAILABLE:
            qr_file = os.path.join(".", f"qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            qr_path = generate_qr_code(audio_url, output_file=qr_file, size=150)
            
            if qr_path and os.path.exists(qr_path):
                try:
                    qr_image = RLImage(qr_path, width=40*mm, height=40*mm)
                    metadata_rows.append([
                        create_paragraph("음성 증거 첨부", 'TableLabelStyle'),
                        qr_image
                    ])
                except Exception:
                    metadata_rows.append([
                        create_paragraph("음성 증거 첨부", 'TableLabelStyle'),
                        create_paragraph("QR 코드 생성 실패", 'TableValueStyle')
                    ])
        
        if metadata_rows:
            metadata_table = Table(metadata_rows, colWidths=[50*mm, 110*mm])
            metadata_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(metadata_table)
    
    doc.build(story)

def create_employment_contract_pdf(content, doc_type, info_json, buffer, voice_signature=None):
    """근로계약서 형식의 구조화된 PDF를 생성합니다."""
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25*mm,
        leftMargin=25*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    story = []
    
    # 제목
    story.append(create_paragraph(f"<b>{doc_type}</b>", 'DocTitle'))
    story.append(Spacer(1, 10*mm))
    
    # 당사자 정보 테이블
    party_data = [
        [create_paragraph("<b>구분</b>", 'TableLabelStyle'), 
         create_paragraph("<b>성명(상호)</b>", 'TableLabelStyle'), 
         create_paragraph("<b>주소</b>", 'TableLabelStyle'), 
         create_paragraph("<b>연락처</b>", 'TableLabelStyle')],
        [create_paragraph("근로자", 'TableLabelStyle'), 
         create_paragraph(info_json.get("name", ""), 'TableValueStyle'),
         create_paragraph(info_json.get("address", ""), 'TableValueStyle'), 
         create_paragraph(info_json.get("phone", ""), 'TableValueStyle')],
        [create_paragraph("사용자", 'TableLabelStyle'), 
         create_paragraph(info_json.get("employer", ""), 'TableValueStyle'),
         create_paragraph("", 'TableValueStyle'), 
         create_paragraph("", 'TableValueStyle')],
    ]
    
    party_table = Table(party_data, colWidths=[30*mm, 50*mm, 60*mm, 40*mm])
    party_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), KOREAN_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(party_table)
    story.append(Spacer(1, 10*mm))
    
    # 근로 조건 및 내용
    story.append(create_paragraph("<b>■ 근로 조건 및 내용</b>", 'TableLabelStyle'))
    story.append(Spacer(1, 5*mm))
    
    content_data = [
        [create_paragraph(content, 'ContentStyle')]
    ]
    content_table = Table(content_data, colWidths=[160*mm])
    content_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(content_table)
    story.append(Spacer(1, 15*mm))
    
    # 전자 서명 메타데이터 서명란
    if voice_signature:
        story.append(create_paragraph("<b>■ 전자 서명 및 증거 메타데이터</b>", 'TableLabelStyle'))
        story.append(Spacer(1, 5*mm))
        
        metadata_rows = []
        signer_name = info_json.get("name", "미상")
        metadata_rows.append([
            create_paragraph("전자 서명 주체", 'TableLabelStyle'),
            create_paragraph(f"근로자: {signer_name} (음성 동의 완료)", 'TableValueStyle')
        ])
        
        timestamp = voice_signature.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        metadata_rows.append([
            create_paragraph("전자 서명 일시", 'TableLabelStyle'),
            create_paragraph(timestamp, 'TableValueStyle')
        ])
        
        doc_hash = voice_signature.get("document_hash", "")
        if doc_hash:
            hash_display = f"{doc_hash[:16]}...{doc_hash[-8:]}"
            metadata_rows.append([
                create_paragraph("문서 해시", 'TableLabelStyle'),
                create_paragraph(f"SHA-256: {hash_display}", 'TableValueStyle')
            ])
        
        # QR 코드 생성 및 삽입
        audio_url = get_audio_file_url(voice_signature.get("audio_file_path", ""), use_web_url=True)
        if audio_url and QR_AVAILABLE:
            qr_file = os.path.join(".", f"qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            qr_path = generate_qr_code(audio_url, output_file=qr_file, size=150)
            
            if qr_path and os.path.exists(qr_path):
                try:
                    qr_image = RLImage(qr_path, width=40*mm, height=40*mm)
                    metadata_rows.append([
                        create_paragraph("음성 증거 첨부", 'TableLabelStyle'),
                        qr_image
                    ])
                except Exception:
                    metadata_rows.append([
                        create_paragraph("음성 증거 첨부", 'TableLabelStyle'),
                        create_paragraph("QR 코드 생성 실패", 'TableValueStyle')
                    ])
        
        if metadata_rows:
            metadata_table = Table(metadata_rows, colWidths=[50*mm, 110*mm])
            metadata_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(metadata_table)
    
    doc.build(story)

def create_document_pdf(content, doc_type, info_json, output, voice_signature=None):
    """doc_type에 따라 적절한 PDF 템플릿 함수를 호출합니다.
    
    Args:
        output: BytesIO 버퍼 또는 파일 경로 (문자열)
    """
    if doc_type in ["개인정보 제공 동의서", "주민등록등본 발급 신청서", "주민등록등본 신청서"]:
        create_application_form_pdf(content, doc_type, info_json, output, voice_signature)
    elif doc_type == "근로계약서":
        create_employment_contract_pdf(content, doc_type, info_json, output, voice_signature)
    else:
        # 기본 템플릿
        doc = SimpleDocTemplate(
            output,
            pagesize=A4,
            rightMargin=30*mm,
            leftMargin=30*mm,
            topMargin=30*mm,
            bottomMargin=30*mm
        )
        
        story = []
        story.append(create_paragraph(f"<b>{doc_type}</b>", 'GenericTitle'))
        story.append(Spacer(1, 20*mm))
        
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(create_paragraph(para, 'GenericBody'))
                story.append(Spacer(1, 6))
        
        doc.build(story)

# ==========================================
# [0] 기본 페이지 설정 및 초기화
# ==========================================
st.set_page_config(page_title="Accessible Voice PDF", layout="centered")

st.markdown(
    """
    <style>
    .big-btn { font-size:20px; padding:18px 24px; border-radius:12px; cursor:pointer; }
    .high-contrast { background-color:#0B5FFF; color: #FFFFFF; border:none; }
    .guide-box { background-color:#e8f0fe; padding:15px; border-radius:10px; border: 1px solid #0B5FFF; margin-bottom: 20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("말하는대로") 

# 세션 상태 변수 초기화
if 'plain_text' not in st.session_state:
    st.session_state.plain_text = ""

if 'system_key' not in st.session_state:
    st.session_state.system_key = Fernet.generate_key()

if 'encrypted_text' not in st.session_state:
    st.session_state.encrypted_text = ""

if 'personal_info' not in st.session_state:
    st.session_state.personal_info = None

if 'document_content' not in st.session_state:
    st.session_state.document_content = None

if 'voice_signature' not in st.session_state:
    st.session_state.voice_signature = None

if 'pdf_filepath' not in st.session_state:
    st.session_state.pdf_filepath = None

# 스크린리더 설정
if 'screen_reader_enabled' not in st.session_state:
    st.session_state.screen_reader_enabled = False

# 스크린리더 토글 (상단에 배치)
with st.sidebar:
    st.markdown("### 🔊 접근성 설정")
    screen_reader_enabled = st.checkbox(
        "스크린리더 활성화", 
        value=st.session_state.screen_reader_enabled,
        help="화면의 주요 내용을 자동으로 음성으로 읽어줍니다."
    )
    st.session_state.screen_reader_enabled = screen_reader_enabled
    
    if screen_reader_enabled:
        st.info("✅ 스크린리더가 활성화되었습니다.")
    else:
        st.caption("스크린리더가 비활성화되어 있습니다.")

# ==========================================
# [1단계] 서류 종류 선택
# ==========================================
st.header("[1단계] 서류 종류 선택")

# 스크린리더: 1단계 자동 안내
if st.session_state.screen_reader_enabled:
    if 'step1_announced' not in st.session_state:
        screen_reader_announce_sync("1단계입니다. 작성할 서류 종류를 선택해주세요.")
        st.session_state.step1_announced = True

if st.button("🔊 1단계 안내 듣기"):
    tts_play("1단계입니다. 작성할 서류 종류를 선택해주세요.")

template_options = {
    "선택": {
        "guide": "[📢입력 가이드]\n\n작성할 서류 종류를 선택해주세요.",
        "announcement": ""
    },
    "근로계약서": {
        "guide": "[📢입력 가이드]\n\n이 서류는 '이름', '근무지', '시급', '근무시간' 순서로 말씀해 주세요.\n\n예시: 홍길동, XX수학 학원, 시급 만원, 아침 9시부터 6시까지",
        "announcement": "근로계약서를 선택하였습니다. 이름, 근무지, 시급, 근무시간 순서로 말씀해주세요"
    },
    "주민등록등본 신청서": {
        "guide": "[📢입력 가이드]\n\n이 서류는 '성명', '거주지 주소', '주민등록번호' 순서로 말씀해 주세요.\n\n예시: 오지헌, 대구 북구, 950101-1234567",
        "announcement": "주민등록등본 신청서를 선택하였습니다. 성명, 거주지 주소, 주민등록번호 순서로 말씀해주세요"
    },
    "개인정보 제공 동의서": {
        "guide": "[📢입력 가이드]\n\n이 서류는 '성명', '생년월일', '주소', '연락처' 순서로 말씀해 주세요.\n\n예시: 홍길동, 1990년 1월 1일, 서울시 강남구, 010-1234-5678",
        "announcement": "개인정보 제공 동의서를 선택하였습니다. 성명, 생년월일, 주소, 연락처 순서로 말씀해주세요"
    }
}

# 이전 선택값 저장
if 'previous_template' not in st.session_state:
    st.session_state.previous_template = "선택"

selected_template = st.selectbox("작성할 서류 종류를 선택하세요.", list(template_options.keys()))

# 템플릿 선택 변경 감지 및 스크린리더 안내
if selected_template != st.session_state.previous_template and selected_template != "선택":
    if st.session_state.screen_reader_enabled:
        announcement = template_options[selected_template].get("announcement", "")
        if announcement:
            screen_reader_announce_sync(announcement)
            # 2단계로 자동 스크롤
            scroll_script = """
            <script>
                setTimeout(function() {
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                }, 2000);
            </script>
            """
            st.markdown(scroll_script, unsafe_allow_html=True)
    st.session_state.previous_template = selected_template

st.markdown(f"""<div class="guide-box">{template_options[selected_template]['guide']}</div>""", unsafe_allow_html=True)

# 템플릿 선택 완료 후 2단계 표시
if selected_template != "선택":
    # [2단계] 개인정보 음성 입력
    st.markdown("---")
    st.header("[2단계] 개인정보 음성 입력")

    # 스크린리더: 단계 안내 (템플릿 선택 후에만)
    if st.session_state.screen_reader_enabled:
        if 'step2_announced' not in st.session_state or st.session_state.previous_template == "선택":
            screen_reader_announce_sync("2단계입니다. 개인정보를 음성으로 입력하세요.")
            st.session_state.step2_announced = True

    st.markdown("### 오디오 녹음")
    st.info("💡 마이크 버튼을 클릭하여 녹음을 시작하세요. 녹음이 끝나면 다시 버튼을 눌러 중지하세요.")

    # 녹음 상태 표시
    if "audio_buffer_obj" in st.session_state:
        buffer = st.session_state["audio_buffer_obj"]
        if len(buffer._audio_segments) > 0:
            segment_count = len(buffer._audio_segments)
            # AudioSegment의 총 길이로 녹음 시간 계산
            total_audio = buffer.to_pydub_audiosegment()
            if len(total_audio) > 0:
                duration_seconds = len(total_audio) / 1000.0  # pydub은 밀리초 단위
                st.caption(f"🎤 녹음 중... 세그먼트: {segment_count}, 녹음 시간: {duration_seconds:.1f}초")

    save_frames_from_audio_receiver(wavpath)

    # 녹음된 파일이 있으면 재생
    if Path(wavpath).exists():
        st.markdown(f"**녹음 파일:** {wavpath}")
        display_wavfile(wavpath)
        
        # Whisper 변환 버튼
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎤 Whisper로 텍스트 변환", key="whisper_convert", help="녹음된 오디오를 텍스트로 변환합니다."):
                with st.spinner("Whisper 모델 로딩 및 변환 중..."):
                    try:
                        if "whisper_model" not in st.session_state:
                            st.session_state.whisper_model = whisper.load_model("small")
                        model = st.session_state.whisper_model
                        result = model.transcribe(str(wavpath))
                        transcribed_text = result["text"]
                        st.session_state["voice_text"] = transcribed_text
                        st.success("✅ 변환 완료")
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync(f"음성이 텍스트로 변환되었습니다. {transcribed_text}")
                    except Exception as e:
                        st.error(f"❌ 변환 중 오류 발생: {str(e)}")
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync("변환 중 오류가 발생했습니다.")
        with col2:
            if st.button("🔄 녹음 초기화", key="reset_recording", help="녹음을 초기화합니다."):
                if "audio_buffer_obj" in st.session_state:
                    st.session_state["audio_buffer_obj"].clear()
                cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                st.session_state["wavpath"] = str(TMP_DIR / f"{cur_time}.wav")
                st.rerun()

    # 음성에서 가져온 텍스트 표시
    st.markdown("### 음성에서 가져온 텍스트")
    if st.session_state.get("voice_text"):
        st.text_area("Recognized text (from voice)", value=st.session_state.get("voice_text", ""), key="voice_text", height=140, label_visibility="collapsed")
    else:
        st.text_area("Recognized text (from voice)", value="", key="voice_text", height=140, label_visibility="collapsed",
                     help="위의 녹음 후 'Whisper로 텍스트 변환' 버튼을 누르거나 직접 입력하세요.")

    input_text = st.text_area("📝 직접 입력:", height=100, help="입력 후 '개인정보 추출' 버튼을 눌러주세요.")

    # 개인정보 추출 버튼
    if st.button("🔍 개인정보 추출하기", type="primary", use_container_width=True):
        voice_text = st.session_state.get("voice_text", "")
        
        # 텍스트 결합: voice_text와 input_text를 합침 (둘 다 있으면 공백으로 구분)
        combined_text = ""
        if voice_text and input_text:
            combined_text = f"{voice_text} {input_text}"
        elif voice_text:
            combined_text = voice_text
        elif input_text:
            combined_text = input_text
        
        if not combined_text:
            st.warning("⚠️ 텍스트를 입력하거나 음성을 변환해주세요.")
        
        else:
            with st.spinner("개인정보 추출 중..."):
                try:
                    personal_info = extract_personal_info(combined_text)
                    st.session_state.personal_info = personal_info
                    st.session_state.plain_text = combined_text
                    
                    # 자동 암호화
                    cipher = Fernet(st.session_state.system_key)
                    encrypted_bytes = cipher.encrypt(combined_text.encode())
                    st.session_state.encrypted_text = encrypted_bytes.decode()
                    
                    st.success("✅ 개인정보 추출 완료!")
                    st.json(personal_info)
                    
                    # 스크린리더 안내
                    if st.session_state.screen_reader_enabled:
                        name = personal_info.get("name", "이름 없음")
                        screen_reader_announce_sync(f"개인정보 추출이 완료되었습니다. 이름: {name}")
                    
                    # 문서 생성
                    with st.spinner("문서 내용 생성 중..."):
                        document_content = generate_document_content(personal_info, selected_template)
                        st.session_state.document_content = document_content
                        st.success("✅ 문서 내용 생성 완료!")
                        
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync("문서 내용이 생성되었습니다. 3단계에서 확인하실 수 있습니다.")
                        
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")

# ==========================================
# [3단계] 서류 확인 및 PDF 생성
# ==========================================
# 템플릿 선택 후에만 3단계 표시
if selected_template != "선택":
    st.markdown("---")
    st.header("[3단계] 서류 확인 및 다운로드")

    # 스크린리더: 단계 안내
    if st.session_state.screen_reader_enabled:
        if 'step3_announced' not in st.session_state and st.session_state.document_content:
            screen_reader_announce_sync("3단계입니다. 생성된 문서를 확인하고 PDF를 생성하세요.")
            st.session_state.step3_announced = True

    if st.button("🔊 3단계 안내 듣기"):
        tts_play("3단계입니다. 생성된 문서를 확인하고, PDF 생성 버튼을 눌러 서류를 다운로드하세요.")

    if not st.session_state.document_content:
        st.info("☝️ 위 2단계에서 개인정보를 추출하고 문서를 생성해주세요.")
    else:
        st.caption("📄 생성된 문서 내용:")
        st.text_area("문서 내용", value=st.session_state.document_content, height=200, disabled=True)

        # 파일 저장 옵션
        save_to_file = st.checkbox("💾 파일로 저장하기", value=False, help="PDF를 로컬 파일로 저장합니다.")
        output_dir = "documents" if save_to_file else None
        
        # PDF 생성 버튼
        if st.button("📄 PDF 서류 생성하기", type="primary", use_container_width=True):
            if not st.session_state.personal_info or not st.session_state.document_content:
                st.error("PDF로 만들 데이터가 없습니다.")
            else:
                try:
                    if save_to_file:
                        # 파일로 저장
                        document_content, filepath = generate_document(
                            st.session_state.personal_info,
                            selected_template,
                            save_file=True,
                            output_dir=output_dir
                        )
                        st.session_state.pdf_filepath = filepath
                        
                        # 파일 내용 읽기
                        with open(filepath, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        st.success(f"✅ PDF 생성 및 저장 완료! 파일: {filepath}")
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync(f"PDF가 생성되어 저장되었습니다. 다운로드 버튼을 눌러 다운로드하실 수 있습니다.")
                        st.download_button(
                            "📥 PDF 다운로드", 
                            data=pdf_bytes, 
                            file_name=os.path.basename(filepath), 
                            mime="application/pdf", 
                            use_container_width=True
                        )
                    else:
                        # 메모리 버퍼로 생성
                        buffer = BytesIO()
                        create_document_pdf(
                            st.session_state.document_content,
                            selected_template,
                            st.session_state.personal_info,
                            buffer,
                            voice_signature=st.session_state.voice_signature
                        )
                        buffer.seek(0)

                        st.success("✅ PDF 생성 완료!")
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync("PDF가 생성되었습니다. 다운로드 버튼을 눌러 다운로드하실 수 있습니다.")
                        st.download_button(
                            "📥 PDF 다운로드", 
                            data=buffer.getvalue(), 
                            file_name=f"{selected_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                            mime="application/pdf", 
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"PDF 생성 중 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # ==========================================
        # [4단계] 음성 서명 (선택)
        # ==========================================
        st.markdown("---")
        st.header("[4단계] 음성 서명 (선택)")
        
        # 스크린리더: 단계 안내
        if st.session_state.screen_reader_enabled:
            if 'step4_announced' not in st.session_state:
                screen_reader_announce_sync("4단계입니다. 선택 사항으로 음성 서명을 추가할 수 있습니다.")
                st.session_state.step4_announced = True
        
        use_voice_signature = st.checkbox("🎤 음성 서명 사용하기", value=False, help="음성 서명을 PDF에 포함시킵니다.")
        
        # 스크린리더: 체크박스 상태 안내
        if st.session_state.screen_reader_enabled and use_voice_signature:
            if 'voice_signature_checked' not in st.session_state:
                screen_reader_announce_sync("음성 서명이 활성화되었습니다. 동의 문구를 녹음하세요.")
                st.session_state.voice_signature_checked = True
        
        if use_voice_signature:
            st.markdown("### 음성 동의 녹음")
            st.info("💡 '본인은 상기 내용을 확인하고 이에 동의합니다.' 라고 말씀해주세요. 마이크 버튼을 클릭하여 녹음을 시작하세요.")
            
            # 음성 서명용 녹음 경로
            if "signature_wavpath" not in st.session_state:
                cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                st.session_state["signature_wavpath"] = str(TMP_DIR / f"signature_{cur_time}.wav")
            
            signature_wavpath = st.session_state["signature_wavpath"]
            
            # 녹음 상태 표시
            if "signature_audio_buffer_obj" in st.session_state:
                buffer = st.session_state["signature_audio_buffer_obj"]
                if len(buffer._audio_segments) > 0:
                    segment_count = len(buffer._audio_segments)
                    # AudioSegment의 총 길이로 녹음 시간 계산
                    total_audio = buffer.to_pydub_audiosegment()
                    if len(total_audio) > 0:
                        duration_seconds = len(total_audio) / 1000.0  # pydub은 밀리초 단위
                        st.caption(f"🎤 음성 서명 녹음 중... 세그먼트: {segment_count}, 녹음 시간: {duration_seconds:.1f}초")
            
            # 음성 서명용 별도 녹음 (기존 녹음과 분리)
            def save_signature_audio(wavpath):
                # 세션 상태 초기화
                if "signature_audio_buffer_obj" not in st.session_state:
                    st.session_state["signature_audio_buffer_obj"] = AudioFrameBuffer()

                buffer = st.session_state["signature_audio_buffer_obj"]
                
                webrtc_ctx = webrtc_streamer(
                    key="signature-audio",
                    mode=WebRtcMode.SENDONLY,
                    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
                    audio_processor_factory=lambda: AudioProcessor(buffer),
                )

                # 녹음이 끝나면 버퍼를 WAV로 저장
                if webrtc_ctx.state.playing is False and len(buffer._audio_segments) > 0:
                    if buffer.to_wav_file(wavpath):
                        buffer.clear()
                        st.success("음성 서명 녹음이 완료되었습니다.")
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync("음성 서명 녹음이 완료되었습니다. 음성 서명 생성 버튼을 눌러주세요.")
            
            save_signature_audio(signature_wavpath)
            
            if Path(signature_wavpath).exists():
                st.markdown(f"**음성 서명 파일:** {signature_wavpath}")
                display_wavfile(signature_wavpath)
                
                if st.button("✅ 음성 서명 생성", type="primary"):
                    # 음성 서명 검증: 동의 문구 확인
                    target_phrase = "본인은 상기 내용을 확인하고 이에 동의합니다."
                    with st.spinner("음성 서명 검증 중... (동의 문구 확인)"):
                        similarity, transcribed_text, is_valid = verify_consent_phrase(
                            signature_wavpath, 
                            target_phrase=target_phrase, 
                            threshold=0.6
                        )
                    
                    if not is_valid:
                        st.error(f"❌ 동의 문구가 확인되지 않았습니다.")
                        if transcribed_text:
                            st.warning(f"**인식된 텍스트:** {transcribed_text}")
                            if similarity is not None:
                                st.warning(f"**유사도:** {similarity*100:.1f}% (필요: 60% 이상)")
                            st.info(f"💡 다음 문구를 정확히 말씀해주세요: \"{target_phrase}\"")
                        else:
                            st.warning("음성을 텍스트로 변환할 수 없습니다. 다시 녹음해주세요.")
                        st.stop()  # 검증 실패 시 진행 중단
                    
                    # 검증 통과
                    if similarity is not None:
                        st.success(f"✅ 동의 문구 확인 완료! (유사도: {similarity*100:.1f}%)")
                        if transcribed_text:
                            st.caption(f"인식된 텍스트: \"{transcribed_text}\"")
                        # 스크린리더 안내
                        if st.session_state.screen_reader_enabled:
                            screen_reader_announce_sync(f"동의 문구가 확인되었습니다. 유사도 {similarity*100:.1f}퍼센트입니다.")
                    
                    if not st.session_state.pdf_filepath:
                        # 임시로 PDF 파일 생성
                        if not os.path.exists("documents"):
                            os.makedirs("documents")
                        temp_pdf = os.path.join("documents", f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                        with open(temp_pdf, 'wb') as f:
                            create_document_pdf(
                                st.session_state.document_content,
                                selected_template,
                                st.session_state.personal_info,
                                f,
                                voice_signature=None
                            )
                        st.session_state.pdf_filepath = temp_pdf
                
                try:
                    voice_signature = create_voice_signature(
                        st.session_state.document_content,
                        st.session_state.pdf_filepath,
                        signature_wavpath
                    )
                    
                    # S3 업로드 옵션
                    upload_to_s3 = st.checkbox("☁️ S3에 오디오 업로드", value=False, help="음성 파일을 AWS S3에 업로드합니다.")
                    if upload_to_s3:
                        audio_url = upload_audio_to_s3(signature_wavpath)
                        if audio_url:
                            voice_signature["audio_file_url"] = audio_url
                            st.success(f"✅ S3 업로드 완료: {audio_url}")
                        else:
                            st.warning("⚠️ S3 업로드 실패 (환경 변수 확인 필요)")
                    
                    st.session_state.voice_signature = voice_signature
                    
                    # 음성 서명 저장
                    signature_file = save_voice_signature(voice_signature, output_dir="documents")
                    st.success(f"✅ 음성 서명 생성 완료! 서명 데이터: {signature_file}")
                    st.json(voice_signature)
                    
                    # 스크린리더 안내
                    if st.session_state.screen_reader_enabled:
                        screen_reader_announce_sync("음성 서명이 생성되었습니다. PDF에 포함되어 재생성됩니다.")
                    
                    # 음성 서명이 포함된 PDF 재생성
                    if st.session_state.pdf_filepath:
                        with st.spinner("음성 서명이 포함된 PDF 재생성 중..."):
                            backup_filepath = st.session_state.pdf_filepath.replace('.pdf', '_backup.pdf')
                            if os.path.exists(st.session_state.pdf_filepath):
                                import shutil
                                shutil.copy2(st.session_state.pdf_filepath, backup_filepath)
                            
                            with open(st.session_state.pdf_filepath, 'wb') as f:
                                create_document_pdf(
                                    st.session_state.document_content,
                                    selected_template,
                                    st.session_state.personal_info,
                                    f,
                                    voice_signature=voice_signature
                                )
                            
                            # 해시값 업데이트
                            new_hash = calculate_document_hash(st.session_state.pdf_filepath)
                            if new_hash:
                                voice_signature["document_hash"] = new_hash
                                save_voice_signature(voice_signature, output_dir="documents")
                            
                            st.success("✅ 음성 서명이 포함된 PDF가 재생성되었습니다!")
                            
                            # 스크린리더 안내
                            if st.session_state.screen_reader_enabled:
                                screen_reader_announce_sync("음성 서명이 포함된 PDF가 재생성되었습니다. 다운로드 버튼을 눌러 다운로드하실 수 있습니다.")
                            
                            # 재생성된 PDF 다운로드
                            with open(st.session_state.pdf_filepath, 'rb') as f:
                                pdf_bytes = f.read()
                            st.download_button(
                                "📥 음성 서명 포함 PDF 다운로드",
                                data=pdf_bytes,
                                file_name=os.path.basename(st.session_state.pdf_filepath),
                                mime="application/pdf",
                                use_container_width=True
                            )
                    
                except Exception as e:
                    st.error(f"음성 서명 생성 중 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())