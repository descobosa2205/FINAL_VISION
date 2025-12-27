# hand_sequence_then_paint.py
# Requiere: pip install opencv-python mediapipe numpy

import cv2
import time
import numpy as np
import mediapipe as mp

# -----------------------------
# Finger counting (MediaPipe)
# -----------------------------
TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP_IDS = {"thumb": 2}


def count_fingers(hand_landmarks, handedness_label: str) -> int:
    lm = hand_landmarks.landmark
    fingers_up = 0

    # Pulgar: tip.x vs mcp.x según mano
    thumb_tip = lm[TIP_IDS["thumb"]]
    thumb_mcp = lm[MCP_IDS["thumb"]]
    if handedness_label == "Right":
        if thumb_tip.x < thumb_mcp.x:
            fingers_up += 1
    else:  # Left
        if thumb_tip.x > thumb_mcp.x:
            fingers_up += 1

    # Índice/medio/anular/meñique: tip.y < pip.y
    for name in ["index", "middle", "ring", "pinky"]:
        tip = lm[TIP_IDS[name]]
        pip = lm[PIP_IDS[name]]
        if tip.y < pip.y:
            fingers_up += 1

    return fingers_up


def landmark_to_pixel(landmark, w: int, h: int):
    return int(landmark.x * w), int(landmark.y * h)


def inside_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)


# -----------------------------
# Reverse sequence detector
# -----------------------------
class ReverseSequenceDetector:
    def __init__(
        self,
        sequence=(5, 4, 3, 2, 1, 0),
        stable_frames=6,          # frames seguidos para aceptar un valor
        step_timeout_s=3.0,       # si no avanza en X s, reinicia
        cooldown_s=1.5
    ):
        self.sequence = list(sequence)
        self.stable_frames = int(stable_frames)
        self.step_timeout_s = float(step_timeout_s)
        self.cooldown_s = float(cooldown_s)

        self.idx = 0
        self.last_count = None
        self.stable = 0
        self.last_step_time = time.time()
        self.cooldown_until = 0.0

    def reset(self):
        self.idx = 0
        self.last_count = None
        self.stable = 0
        self.last_step_time = time.time()

    def update(self, count):
        """
        count: int 0..5 o None si no hay mano
        return: True si se completa la secuencia
        """
        now = time.time()

        if now < self.cooldown_until:
            return False

        if count is None:
            return False

        if (now - self.last_step_time) > self.step_timeout_s:
            self.reset()

        if self.last_count == count:
            self.stable += 1
        else:
            self.last_count = count
            self.stable = 1

        if self.stable < self.stable_frames:
            return False

        expected = self.sequence[self.idx]

        if count == expected:
            self.idx += 1
            self.last_step_time = now
            self.stable = 0

            if self.idx >= len(self.sequence):
                print("✅ Secuencia 5→4→3→2→1→0 detectada correctamente. Entrando en modo pintura…")
                self.cooldown_until = now + self.cooldown_s
                self.reset()
                return True
            return False

        # Atajo: si vuelve a mostrar 5, reanclar desde el inicio
        if count == self.sequence[0]:
            self.idx = 1
            self.last_step_time = now
            self.stable = 0
            return False

        return False


# -----------------------------
# App (secuencia -> pintar)
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    MODE_SEQUENCE = 0
    MODE_PAINT = 1
    mode = MODE_SEQUENCE

    # Secuencia
    seq = ReverseSequenceDetector(sequence=(5, 4, 3, 2, 1, 0), stable_frames=6, step_timeout_s=3.0, cooldown_s=1.5)

    # Pintura
    canvas = None
    ema_alpha = 0.35
    ema_x, ema_y = None, None
    prev_draw_point = None

    HOLD_SECONDS = 3.0
    hold_start_time = None
    hold_target = None  # 0/2/3/4/5

    # Colores (BGR)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    current_color = COLOR_RED

    brush_thickness = 6
    desired_width = 1280  # baja a 640 si va lento

    # FPS
    prev_fps_t = time.time()
    fps = 0.0

    def reset_to_sequence():
        nonlocal mode, canvas, ema_x, ema_y, prev_draw_point, hold_start_time, hold_target, current_color
        print("🔄 Reiniciando: volviendo a detección de secuencia…")
        mode = MODE_SEQUENCE
        seq.reset()
        canvas = None
        ema_x = ema_y = None
        prev_draw_point = None
        hold_start_time = None
        hold_target = None
        current_color = COLOR_RED

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS (cálculo simple por frame)
            now_t = time.time()
            dt = now_t - prev_fps_t
            if dt > 0:
                fps = 1.0 / dt
            prev_fps_t = now_t

            frame = cv2.flip(frame, 1)

            # Resize opcional
            h0, w0 = frame.shape[:2]
            if desired_width is not None and w0 != desired_width:
                scale = desired_width / w0
                frame = cv2.resize(frame, (desired_width, int(h0 * scale)))
            h, w = frame.shape[:2]

            # Procesar mano
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            fingers_count = None
            hand_label = None
            index_tip_px = None
            hand_landmarks = None

            if res.multi_hand_landmarks and res.multi_handedness:
                hand_landmarks = res.multi_hand_landmarks[0]
                hand_label = res.multi_handedness[0].classification[0].label  # "Left"/"Right"
                fingers_count = count_fingers(hand_landmarks, hand_label)

                # índice tip para modo pintura
                tip = hand_landmarks.landmark[TIP_IDS["index"]]
                x, y = landmark_to_pixel(tip, w, h)

                # EMA
                if ema_x is None:
                    ema_x, ema_y = x, y
                else:
                    ema_x = int(ema_alpha * x + (1 - ema_alpha) * ema_x)
                    ema_y = int(ema_alpha * y + (1 - ema_alpha) * ema_y)
                index_tip_px = (ema_x, ema_y)

            # -----------------------------
            # MODO: SECUENCIA
            # -----------------------------
            if mode == MODE_SEQUENCE:
                # UI
                title = "SECUENCIA (5-4-3-2-1-0)  |  ESPACIO: salir  |  R: reiniciar"
                cv2.putText(frame, "MUESTRA LA SECUENCIA: 5,4,3,2,1,0", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                if hand_landmarks is not None:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                expected = seq.sequence[seq.idx] if seq.idx < len(seq.sequence) else "-"
                cv2.putText(frame, f"Dedos: {fingers_count if fingers_count is not None else '-'}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Paso: {seq.idx}/{len(seq.sequence)}  (espera {expected})", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                completed = seq.update(fingers_count)
                if completed:
                    # entrar a modo pintura
                    mode = MODE_PAINT
                    canvas = np.zeros_like(frame)
                    prev_draw_point = None
                    hold_start_time = None
                    hold_target = None
                    # (dejamos current_color como esté)
                    ema_x, ema_y = None, None  # reset suave al entrar
                    print("🖊️ Modo pintura activado.")

                out = frame

            # -----------------------------
            # MODO: PINTAR
            # -----------------------------
            else:
                title = "PINTAR  |  ESPACIO: salir  |  R: reiniciar"

                if canvas is None:
                    canvas = np.zeros_like(frame)

                # Zona de dibujo (recuadro grande)
                margin_x = int(0.06 * w)
                margin_y_top = int(0.14 * h)
                margin_y_bot = int(0.06 * h)
                draw_rect = (margin_x, margin_y_top, w - margin_x, h - margin_y_bot)

                # Leyenda fija
                legend_lines = [
                    "INSTRUCCIONES:",
                    "Pinta con el indice dentro del recuadro.",
                    "0 dedos (mano cerrada) 3s = borrar",
                    "5 dedos 3s = ROJO | 4 dedos 3s = AZUL",
                    "3 dedos 3s = VERDE | 2 dedos 3s = AMARILLO",
                    "ESPACIO = salir | R = reiniciar",
                ]
                legend_x, legend_y = 20, 30
                pad = 8
                line_h = 22
                box_w = 760
                box_h = pad * 2 + line_h * len(legend_lines)
                cv2.rectangle(frame, (legend_x - 10, legend_y - 25),
                              (legend_x - 10 + box_w, legend_y - 25 + box_h), (0, 0, 0), -1)
                cv2.rectangle(frame, (legend_x - 10, legend_y - 25),
                              (legend_x - 10 + box_w, legend_y - 25 + box_h), (255, 255, 255), 1)
                for i, t in enumerate(legend_lines):
                    cv2.putText(frame, t, (legend_x, legend_y + i * line_h),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Recuadro de dibujo
                x1, y1, x2, y2 = draw_rect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Mostrar color actual (sin timer)
                cv2.putText(frame, "Color actual", (w - 230, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (w - 80, 18), (w - 20, 58), current_color, -1)
                cv2.rectangle(frame, (w - 80, 18), (w - 20, 58), (255, 255, 255), 1)

                if hand_landmarks is not None:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Indicador índice como tu referencia
                    if index_tip_px is not None:
                        cv2.circle(frame, index_tip_px, 10, (0, 255, 0), -1)
                        cv2.circle(frame, index_tip_px, 14, current_color, 2)

                    fingers = fingers_count if fingers_count is not None else None

                    # 1) Gestos mantenidos (interno)
                    if fingers in (0, 2, 3, 4, 5):
                        if hold_target != fingers:
                            hold_target = fingers
                            hold_start_time = time.time()

                        elapsed = time.time() - (hold_start_time or time.time())
                        if elapsed >= HOLD_SECONDS:
                            if hold_target == 0:
                                canvas[:] = 0
                                print("Pantalla limpiada (mano cerrada 3s).")
                            elif hold_target == 5:
                                current_color = COLOR_RED
                                print("Color cambiado: ROJO (5 dedos 3s).")
                            elif hold_target == 4:
                                current_color = COLOR_BLUE
                                print("Color cambiado: AZUL (4 dedos 3s).")
                            elif hold_target == 3:
                                current_color = COLOR_GREEN
                                print("Color cambiado: VERDE (3 dedos 3s).")
                            elif hold_target == 2:
                                current_color = COLOR_YELLOW
                                print("Color cambiado: AMARILLO (2 dedos 3s).")

                            hold_target = None
                            hold_start_time = None
                    else:
                        hold_target = None
                        hold_start_time = None

                    # 2) DIBUJO: siempre que haya mano y fingers>=1 (no se detiene con 2/3/4/5)
                    if fingers is not None and fingers >= 1 and index_tip_px is not None:
                        if inside_rect(index_tip_px, draw_rect):
                            if prev_draw_point is None:
                                prev_draw_point = index_tip_px
                            else:
                                if not inside_rect(prev_draw_point, draw_rect):
                                    prev_draw_point = index_tip_px
                                else:
                                    cv2.line(canvas, prev_draw_point, index_tip_px,
                                             current_color, brush_thickness, cv2.LINE_AA)
                                    prev_draw_point = index_tip_px
                        else:
                            prev_draw_point = None
                    else:
                        prev_draw_point = None

                    cv2.putText(frame, f"Dedos: {fingers if fingers is not None else '-'}  Mano: {hand_label}",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Sin mano
                    prev_draw_point = None
                    ema_x, ema_y = None, None
                    hold_target = None
                    hold_start_time = None
                    cv2.putText(frame, "Sin mano detectada", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

                # Superponer canvas sobre frame
                gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                fg = cv2.bitwise_and(canvas, canvas, mask=mask)
                out = cv2.add(bg, fg)

            # FPS
            cv2.putText(
                out,
                f"FPS: {fps:.1f}",
                (w - 160, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(title, out)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # ESPACIO -> cerrar
                break
            if key in (ord("r"), ord("R")):  # reiniciar
                # cerrar la ventana actual para evitar múltiples títulos
                cv2.destroyAllWindows()
                reset_to_sequence()
                continue
            if key == ord("+") and mode == MODE_PAINT:
                brush_thickness = min(30, brush_thickness + 1)
            if key == ord("-") and mode == MODE_PAINT:
                brush_thickness = max(1, brush_thickness - 1)
            if key == ord("c") and mode == MODE_PAINT:
                canvas[:] = 0
                print("Pantalla limpiada (tecla 'c').")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()