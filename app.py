import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import time
from collections import deque

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="PhysioAI - Player Completo",
    page_icon="🩺",
    layout="wide"
)

# --- ESTILO ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #0e1117; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- CLASSE ANALISADORA ---
class AnalisadorADMWeb:
    def __init__(self, id_paciente, tipo_movimento, lado_do_corpo):
        self.id_paciente = id_paciente
        self.tipo_movimento = tipo_movimento
        self.lado_do_corpo = lado_do_corpo.lower()
        self.lista_angulos_adm = []
        self.historico_angulos = deque(maxlen=5)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self._configurar_movimento()

    def _configurar_movimento(self):
        if self.lado_do_corpo == 'esquerdo':
            p_ombro = self.mp_pose.PoseLandmark.LEFT_SHOULDER
            p_cotovelo = self.mp_pose.PoseLandmark.LEFT_ELBOW
            p_pulso = self.mp_pose.PoseLandmark.LEFT_WRIST
            p_quadril = self.mp_pose.PoseLandmark.LEFT_HIP
            p_indicador = self.mp_pose.PoseLandmark.LEFT_INDEX
        else:
            p_ombro = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            p_cotovelo = self.mp_pose.PoseLandmark.RIGHT_ELBOW
            p_pulso = self.mp_pose.PoseLandmark.RIGHT_WRIST
            p_quadril = self.mp_pose.PoseLandmark.RIGHT_HIP
            p_indicador = self.mp_pose.PoseLandmark.RIGHT_INDEX

        if self.tipo_movimento == 'Flexão de Cotovelo':
            self.ponto_a, self.ponto_b, self.ponto_c = p_ombro, p_cotovelo, p_pulso
        elif self.tipo_movimento == 'Abdução de Ombro':
            self.ponto_a, self.ponto_b, self.ponto_c = p_quadril, p_ombro, p_cotovelo
        elif self.tipo_movimento == 'Flexão de Ombro':
            self.ponto_a, self.ponto_b, self.ponto_c = p_quadril, p_ombro, p_cotovelo
        elif self.tipo_movimento in ['Desvio Radial (Punho)', 'Desvio Ulnar (Punho)']:
            self.ponto_a, self.ponto_b, self.ponto_c = p_cotovelo, p_pulso, p_indicador

    def _calcular_angulo_3d(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        if mag_v1 == 0 or mag_v2 == 0:
            return 0.0
        produto = np.dot(v1, v2)
        cos_theta = np.clip(produto / (mag_v1 * mag_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def processar_video_para_memoria(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_processados = []
        angulos_por_frame = []

        progresso = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            angulo_suavizado = 0.0

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                p_a = landmarks[self.ponto_a.value]
                p_b = landmarks[self.ponto_b.value]
                p_c = landmarks[self.ponto_c.value]

                angulo_base = self._calcular_angulo_3d(p_a, p_b, p_c)

                # Ajuste para punho (desvio de 180)
                if 'Punho' in self.tipo_movimento:
                    angulo_final = abs(180 - angulo_base)
                else:
                    angulo_final = angulo_base

                self.historico_angulos.append(angulo_final)
                self.lista_angulos_adm.append(angulo_final)
                angulo_suavizado = float(np.mean(self.historico_angulos))
                angulos_por_frame.append(angulo_suavizado)

                cv2.putText(
                    image_rgb, f"{int(angulo_suavizado)} deg",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3
                )
            else:
                angulos_por_frame.append(0.0)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image_rgb, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

            frames_processados.append(image_rgb)
            frame_count += 1
            if total_frames > 0:
                progresso.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        progresso.empty()
        return frames_processados, angulos_por_frame

# --- ESTADO DA SESSÃO ---
if 'frames_salvos' not in st.session_state:
    st.session_state.frames_salvos = []
if 'angulos_salvos' not in st.session_state:
    st.session_state.angulos_salvos = []
if 'analise_feita' not in st.session_state:
    st.session_state.analise_feita = False
if 'frame_atual' not in st.session_state:
    st.session_state.frame_atual = 0
if 'ultimo_arquivo' not in st.session_state:
    st.session_state.ultimo_arquivo = None
if 'playing' not in st.session_state:
    st.session_state.playing = False

# --- BARRA LATERAL ---
st.sidebar.title("⚙️ Configurações")
id_paciente = st.sidebar.text_input("ID do Paciente", "PAC-001")
movimento = st.sidebar.selectbox("Tipo de Movimento", [
    "Flexão de Cotovelo", "Abdução de Ombro", "Flexão de Ombro",
    "Desvio Radial (Punho)", "Desvio Ulnar (Punho)"
])
lado = st.sidebar.radio("Lado do Corpo", ["Direito", "Esquerdo"])
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Carregar Vídeo", type=["mp4", "avi", "mov"])

st.title("🩺 PhysioAI: Análise & Player")

# Reset ao mudar arquivo
if uploaded_file:
    if st.session_state.ultimo_arquivo != uploaded_file.name:
        st.session_state.frames_salvos = []
        st.session_state.angulos_salvos = []
        st.session_state.analise_feita = False
        st.session_state.frame_atual = 0
        st.session_state.playing = False
        st.session_state.ultimo_arquivo = uploaded_file.name

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()

    if st.sidebar.button("▶️ Processar Vídeo", key="btn_processar_video"):
        with st.spinner(f"Processando {movimento}..."):
            analisador = AnalisadorADMWeb(id_paciente, movimento, lado)
            frames, angulos = analisador.processar_video_para_memoria(tfile.name)
            st.session_state.frames_salvos = frames
            st.session_state.angulos_salvos = angulos
            st.session_state.analise_feita = True
            st.session_state.frame_atual = 0
            st.session_state.playing = False

    # --- ÁREA DO PLAYER ---
    if st.session_state.analise_feita and len(st.session_state.frames_salvos) > 0:
        st.divider()
        col_player, col_stats = st.columns([2, 1])

        with col_player:
            st.subheader(f"Visualização: {movimento}")
            total_frames = len(st.session_state.frames_salvos)

            # --- CONTROLES DE REPRODUÇÃO ---
            with st.container():
                col_btns, col_slide = st.columns([1, 4])

                with col_btns:
                    label = "⏸️ Pausar" if st.session_state.playing else "▶️ Reproduzir"
                    if st.button(label, key="btn_play_pause"):
                        st.session_state.playing = not st.session_state.playing
                        st.rerun()

                with col_slide:
                    idx = st.slider(
                        "Linha do Tempo",
                        0, total_frames - 1,
                        int(st.session_state.frame_atual),
                        key="slider_timeline"
                    )
                    # Se mexeu no slider, pausa e atualiza frame
                    if idx != st.session_state.frame_atual:
                        st.session_state.frame_atual = int(idx)
                        st.session_state.playing = False
                        st.rerun()

            # --- VISUALIZAÇÃO (IMAGEM) ---
            image_placeholder = st.empty()

            # Botões de ajuste fino
            c_prev, c_center, c_next = st.columns([1, 2, 1])
            if c_prev.button("⏪ -10 Frames", key="btn_prev10"):
                st.session_state.frame_atual = max(0, int(st.session_state.frame_atual) - 10)
                st.session_state.playing = False
                st.rerun()

            if c_next.button("+10 Frames ⏩", key="btn_next10"):
                st.session_state.frame_atual = min(total_frames - 1, int(st.session_state.frame_atual) + 10)
                st.session_state.playing = False
                st.rerun()

            # Reprodução
            if st.session_state.playing:
                i0 = int(st.session_state.frame_atual)
                for i in range(i0, total_frames):
                    image_placeholder.image(st.session_state.frames_salvos[i], use_container_width=True)
                    st.session_state.frame_atual = i
                    time.sleep(0.03)

                    # Para automaticamente ao final
                    if i == total_frames - 1:
                        st.session_state.playing = False

                st.rerun()
            else:
                image_placeholder.image(
                    st.session_state.frames_salvos[int(st.session_state.frame_atual)],
                    use_container_width=True
                )

        with col_stats:
            st.subheader("Resultados")
            idx = int(st.session_state.frame_atual)
            total_frames = len(st.session_state.frames_salvos)
            angulo_frame = float(st.session_state.angulos_salvos[idx])

            st.metric("Ângulo Atual", f"{angulo_frame:.1f}°")
            st.caption(f"Frame: {idx}/{total_frames - 1}")

            validos = [x for x in st.session_state.angulos_salvos if x > 1]
            if validos:
                st.markdown("### Resumo")
                st.write(f"Máx: **{max(validos):.1f}°**")
                st.write(f"Média: **{np.mean(validos):.1f}°**")

                df_chart = pd.DataFrame(st.session_state.angulos_salvos, columns=["Graus"])
                st.line_chart(df_chart)

                df_export = pd.DataFrame(st.session_state.angulos_salvos, columns=["Angulo"])
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar CSV", csv, "analise.csv", "text/csv")
else:
    st.info("👈 Carregue um vídeo na barra lateral.")


