# hand_sequence_reverse.py
import cv2
import time
import mediapipe as mp


import subprocess
import sys
import os

# -----------------------------
# Finger counting (MediaPipe)
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

def count_fingers(hand_landmarks, handedness_label, img_w, img_h) -> int:
    """
    Cuenta dedos levantados:
    - Pulgar: depende de mano izquierda/derecha comparando x del tip vs x del IP.
    - Otros dedos: tip.y < pip.y (más arriba en imagen) => levantado.
    """
    lm = hand_landmarks.landmark

    # Convertimos a píxeles para robustez (aunque las comparaciones sirven en normalizado)
    x = [int(lm[i].x * img_w) for i in range(21)]
    y = [int(lm[i].y * img_h) for i in range(21)]

    fingers = 0

    # Pulgar (tip=4, ip=3)
    # En mano derecha, el pulgar "sale" hacia la izquierda en imagen (tip.x < ip.x)
    # En mano izquierda, al revés (tip.x > ip.x)
    if handedness_label == "Right":
        if x[4] < x[3]:
            fingers += 1
    else:  # "Left"
        if x[4] > x[3]:
            fingers += 1

    # Índice, medio, anular, meñique: tip más arriba que PIP => levantado
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if y[tip] < y[pip]:
            fingers += 1

    return fingers

# -----------------------------
# Reverse sequence detector
# -----------------------------
class ReverseSequenceDetector:
    def __init__(
        self,
        sequence=(5, 4, 3, 2, 1, 0),
        stable_frames=6,          # cuántos frames seguidos debe mantenerse el conteo
        step_timeout_s=3.0,       # si no avanza en X segundos, reinicia
        cooldown_s=1.5            # tras completar, espera para no re-disparar instantáneo
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

    def update(self, count: int | None) -> bool:
        """
        count: número de dedos detectados (0..5). Si no hay mano, pasar None.
        Devuelve True SOLO cuando se completa la secuencia.
        """
        now = time.time()

        # Cooldown tras completar
        if now < self.cooldown_until:
            return False

        # Si no hay mano, no avanzamos pero tampoco “rompemos” necesariamente
        # (si prefieres reiniciar al perder mano, descomenta reset)
        if count is None:
            # self.reset()
            return False

        # Timeout si llevamos demasiado sin avanzar
        if (now - self.last_step_time) > self.step_timeout_s:
            self.reset()

        # Estabilidad de conteo (debounce)
        if self.last_count == count:
            self.stable += 1
        else:
            self.last_count = count
            self.stable = 1

        if self.stable < self.stable_frames:
            return False

        expected = self.sequence[self.idx]

        # Si coincide con lo esperado, avanzamos
        if count == expected:
            self.idx += 1
            self.last_step_time = now
            self.stable = 0  # obliga a estabilizar el siguiente valor

            # ¿completado?

            

            if self.idx >= len(self.sequence):
                # Ejecutar el programa finger_paint.py
                subprocess.Popen(["python", "finger_paint.py"])

                # Opcional: lógica previa de cooldown/reset si la necesitas
                self.cooldown_until = now + self.cooldown_s
                self.reset()

                # Cerrar el programa actual
                sys.exit(0)

            return False

        # Si el usuario vuelve a mostrar 5, reiniciamos “limpio”
        if count == self.sequence[0]:
            self.idx = 1  # ya tenemos el primer paso (5)
            self.last_step_time = now
            self.stable = 0
            return False

        # Si muestra algo distinto, no avanzamos.
        # (Opcional) si quieres reiniciar ante cualquier valor inesperado:
        # self.reset()

        return False

# -----------------------------
# Main
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    seq = ReverseSequenceDetector(sequence=(5, 4, 3, 2, 1, 0), stable_frames=6, step_timeout_s=3.0, cooldown_s=1.5)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            fingers_count = None

            if res.multi_hand_landmarks and res.multi_handedness:
                hand_lms = res.multi_hand_landmarks[0]
                handedness_label = res.multi_handedness[0].classification[0].label  # "Left" / "Right"

                fingers_count = count_fingers(hand_lms, handedness_label, w, h)

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Actualiza secuencia (imprime cuando se completa)
            seq.update(fingers_count)

            # UI
            txt = f"Dedos: {fingers_count if fingers_count is not None else '-'} | Paso: {seq.idx}/{len(seq.sequence)} (espera {seq.sequence[seq.idx] if seq.idx < len(seq.sequence) else '-'})"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand + Reverse Sequence", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()