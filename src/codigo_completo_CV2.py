import cv2
import numpy as np
import time
from collections import deque, Counter
from pathlib import Path

# =========================================================
# Calibración opcional
# =========================================================
def load_calib_npz():
    p = Path("calib.npz")
    if not p.exists():
        print("[WARN] No existe calib.npz. Continuo sin undistort.")
        return None, None
    d = np.load(str(p))
    K = d["K"].astype(np.float32)
    dist = d["dist"].astype(np.float32)
    print("[OK] Calibración cargada desde calib.npz")
    return K, dist


# =========================================================
# CUADRADO DONDE SE PINTA
# =========================================================
def get_roi(frame):
    H, W = frame.shape[:2]
    x1, y1 = int(0.02 * W), int(0.10 * H)
    x2, y2 = int(0.48 * W), int(0.96 * H)
    return x1, y1, x2, y2


# =========================================================
# Trackbars
# =========================================================
def nothing(x):
    pass

def setup_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 360)

    cv2.createTrackbar("Cr_low",  "Controls", 140, 255, nothing)
    cv2.createTrackbar("Cr_high", "Controls", 180, 255, nothing)
    cv2.createTrackbar("Cb_low",  "Controls",  80, 255, nothing)
    cv2.createTrackbar("Cb_high", "Controls", 135, 255, nothing)

    cv2.createTrackbar("Blur",    "Controls",   7, 31,  nothing)  # impar
    cv2.createTrackbar("OpenIt",  "Controls",   2, 6,   nothing)
    cv2.createTrackbar("CloseIt", "Controls",   2, 8,   nothing)

def read_trackbars():
    cr_l = cv2.getTrackbarPos("Cr_low", "Controls")
    cr_h = cv2.getTrackbarPos("Cr_high","Controls")
    cb_l = cv2.getTrackbarPos("Cb_low", "Controls")
    cb_h = cv2.getTrackbarPos("Cb_high","Controls")
    blur = cv2.getTrackbarPos("Blur", "Controls")
    op_it = cv2.getTrackbarPos("OpenIt", "Controls")
    cl_it = cv2.getTrackbarPos("CloseIt","Controls")

    if cr_l > cr_h: cr_l, cr_h = cr_h, cr_l
    if cb_l > cb_h: cb_l, cb_h = cb_h, cb_l
    if blur < 1: blur = 1
    if blur % 2 == 0: blur += 1
    op_it = max(0, op_it)
    cl_it = max(0, cl_it)
    return cr_l, cr_h, cb_l, cb_h, blur, op_it, cl_it


# =========================================================
# Segmentación piel YCrCb 
# =========================================================
def skin_mask_ycrcb_sliders(roi_bgr, cr_l, cr_h, cb_l, cb_h, blur_k, op_it, cl_it):
    blur = cv2.GaussianBlur(roi_bgr, (blur_k, blur_k), 0)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, cr_l, cb_l], dtype=np.uint8)
    upper = np.array([255, cr_h, cb_h], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if op_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=op_it)
    if cl_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cl_it)

    return mask


# =========================================================
# Mano sólida: contorno mayor + fill + recorte de muñeca
# =========================================================
def hand_solid_from_mask(mask):
    H, W = mask.shape[:2]
    min_area = int(0.03 * H * W)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, None

    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [c], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(c)
    cut_y = int(y + 0.88 * h)
    solid[cut_y:, :] = 0

    contours2, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        return None, None
    c2 = max(contours2, key=cv2.contourArea)
    if cv2.contourArea(c2) < min_area:
        return None, None

    return solid, c2


# =========================================================
# Conteo de dedos
# =========================================================
def count_fingers_defects(contour):
    cnt = contour.astype(np.int32)
    eps = 0.01 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, eps, True)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    hull_pts = cv2.convexHull(cnt, returnPoints=True)

    if hull_idx is None or len(hull_idx) < 3 or hull_pts is None or len(hull_pts) < 3:
        return 0, hull_pts, None

    hull_idx = np.sort(hull_idx, axis=0)

    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except cv2.error:
        defects = None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return 0, hull_pts, defects
    cy = int(M["m01"] / M["m00"])

    top = hull_pts[hull_pts[:, 0, 1].argmin()][0]
    tip = (int(top[0]), int(top[1]))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = (h / float(w + 1e-6))
    tip_above = (cy - tip[1])

    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull_pts) if hull_pts is not None else area
    solidity = float(area) / float(hull_area + 1e-6)

    def looks_like_one_finger():
        cond_tip = tip_above > 0.12 * h
        cond_shape = aspect > 0.95
        cond_solid = solidity > 0.82
        return cond_tip and (cond_shape or cond_solid)

    if defects is None or defects.shape[0] == 0:
        return (1 if looks_like_one_finger() else 0), hull_pts, defects

    gaps = 0
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = cnt[s][0]
        end   = cnt[e][0]
        far   = cnt[f][0]
        depth = depth / 256.0

        if depth < 8:
            continue

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b < 1e-6 or c < 1e-6:
            continue

        cosang = (b*b + c*c - a*a) / (2*b*c)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))

        if ang < 95:
            gaps += 1

    if gaps == 0:
        return (1 if looks_like_one_finger() else 0), hull_pts, defects

    fingers = min(gaps + 1, 5)
    return fingers, hull_pts, defects


# =========================================================
# Punta para pintar: punto más alto del hull
# =========================================================
def tip_from_hull(hull_pts):
    if hull_pts is None or len(hull_pts) == 0:
        return None
    top = hull_pts[hull_pts[:, 0, 1].argmin()][0]
    return (int(top[0]), int(top[1]))


# =========================================================
# Estabilidad: MODA en ventana
# =========================================================
def stable_mode(history, window=12, require=8):
    if len(history) < window:
        return None
    w = list(history)[-window:]
    m = Counter(w).most_common(1)[0][0]
    return m if w.count(m) >= require else None


# =========================================================
# FSM contraseña
# =========================================================
class PasswordFSM:
    def __init__(self, pattern, step_timeout=3.0):
        self.pattern = pattern
        self.step_timeout = step_timeout
        self.reset()

    def reset(self):
        self.idx = 0
        self.unlocked = False
        self.last_ok = time.time()

    def update(self, stable):
        if self.unlocked or stable is None:
            return
        now = time.time()
        if now - self.last_ok > self.step_timeout:
            self.idx = 0
        if stable == self.pattern[self.idx]:
            self.idx += 1
            self.last_ok = now
            if self.idx >= len(self.pattern):
                self.unlocked = True


# =========================================================
# HUD helpers
# =========================================================
def draw_panel(img, x, y, w, h, alpha=0.55):
    x = max(0, x); y = max(0, y)
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)
    if x >= x2 or y >= y2:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text(img, text, org, scale=0.75, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_password_bar(img, pattern, idx, unlocked, x, y):
    box_w, box_h, gap = 28, 18, 6
    for i, v in enumerate(pattern):
        bx = x + i * (box_w + gap)
        by = y
        if unlocked or i < idx:
            col = (0, 200, 0)
        elif i == idx:
            col = (0, 200, 200)
        else:
            col = (80, 80, 80)
        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), col, -1)
        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), 2)
        cv2.putText(img, str(v), (bx + 8, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

def draw_progress_bar(img, x, y, w, h, progress, label=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill = int(w * np.clip(progress, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + h), (0, 200, 0), -1)
    if label:
        put_text(img, label, (x, y - 6), 0.55, 1)

def draw_inset_mask(out_bgr, mask, title="hand solid", pos="bottom_right", size=(260, 190), margin=18):
    H, W = out_bgr.shape[:2]
    iw, ih = size

    if mask is None:
        mask = np.zeros((ih, iw), dtype=np.uint8)
    else:
        mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    inset = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if pos == "bottom_right":
        x1 = W - iw - margin
        y1 = H - ih - margin
    elif pos == "bottom_left":
        x1 = margin
        y1 = H - ih - margin
    else:
        x1 = W - iw - margin
        y1 = H - ih - margin

    x2, y2 = x1 + iw, y1 + ih

    draw_panel(out_bgr, x1 - 8, y1 - 32, iw + 16, ih + 40, alpha=0.35)
    cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
    put_text(out_bgr, f"Mask: {title}", (x1, y1 - 10), 0.60, 2)
    out_bgr[y1:y2, x1:x2] = inset


# =========================================================
# MAIN
# =========================================================
def main(cam_index=0, width=1280, height=720):
    K, dist = load_calib_npz()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    setup_trackbars()

    pattern = [5, 4, 3, 2, 1]
    fsm = PasswordFSM(pattern, step_timeout=3.0)
    hist = deque(maxlen=30)

    canvas = None
    prev_pt = None

    sm_tip = None
    tip_alpha = 0.75

    # Trazo (ajustable con + / -)
    paint_thickness = 3
    min_th, max_th = 1, 30

    # Color actual (BGR)
    paint_color = (255, 255, 255)  # blanco

    # Borrado por gesto: 5 dedos 3s (unlocked)
    clear_hold_s = 3.0
    clear_timer_start = None
    clear_armed = True

    # Cambio de color por gesto: 4/3/2 dedos 3s (unlocked)
    color_hold_s = 3.0
    color_timer_start = None
    color_target = None
    color_armed = True

    color_map = {
        4: ((255, 0, 0), "AZUL"),
        3: ((0, 0, 255), "ROJO"),
        2: ((0, 255, 0), "VERDE"),
    }

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if K is not None and dist is not None:
            frame = cv2.undistort(frame, K, dist)

        # ====== ESPEJO ======
        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        x1, y1, x2, y2 = get_roi(frame)
        roi = frame[y1:y2, x1:x2].copy()

        cr_l, cr_h, cb_l, cb_h, blur_k, op_it, cl_it = read_trackbars()
        raw_mask = skin_mask_ycrcb_sliders(roi, cr_l, cr_h, cb_l, cb_h, blur_k, op_it, cl_it)
        solid, contour = hand_solid_from_mask(raw_mask)

        finger_count = 0
        tip_global = None
        area_ratio = float(np.count_nonzero(raw_mask)) / float(raw_mask.size + 1e-6)

        if solid is not None and contour is not None and area_ratio < 0.60:
            fingers, hull_pts, _ = count_fingers_defects(contour)
            finger_count = int(fingers)
            hist.append(finger_count)

            tip = tip_from_hull(hull_pts)
            if tip is not None:
                tip_global = (tip[0] + x1, tip[1] + y1)
        else:
            hist.clear()
            prev_pt = None
            sm_tip = None
            finger_count = 0

        stable = stable_mode(hist, window=12, require=8)
        fsm.update(stable)

        # Suavizado del tip
        tip_draw = None
        if tip_global is not None:
            if sm_tip is None:
                sm_tip = np.array(tip_global, dtype=np.float32)
            else:
                sm_tip = tip_alpha * sm_tip + (1 - tip_alpha) * np.array(tip_global, dtype=np.float32)
            tip_draw = (int(sm_tip[0]), int(sm_tip[1]))

        # Pintar (unlocked + stable==1)
        if fsm.unlocked and stable == 1 and tip_draw is not None:
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, tip_draw, paint_color, paint_thickness, cv2.LINE_AA)
            prev_pt = tip_draw
        else:
            prev_pt = None

        # BORRADO: 5 dedos 3s
        clear_progress = None
        if fsm.unlocked and stable == 5 and clear_armed:
            if clear_timer_start is None:
                clear_timer_start = time.time()
            elapsed = time.time() - clear_timer_start
            clear_progress = min(1.0, elapsed / clear_hold_s)
            if elapsed >= clear_hold_s:
                canvas[:] = 0
                clear_timer_start = None
                clear_armed = False
        else:
            clear_timer_start = None
            if stable != 5:
                clear_armed = True

        # CAMBIO COLOR: 4/3/2 dedos 3s
        color_progress = None
        color_label = None
        if fsm.unlocked and stable in (2, 3, 4) and stable != 5:
            if color_target != stable:
                color_target = stable
                color_timer_start = time.time()
                color_armed = True

            if color_armed:
                elapsed = time.time() - (color_timer_start or time.time())
                color_progress = min(1.0, elapsed / color_hold_s)
                color_label = f"Cambiar a {color_map[stable][1]}"
                if elapsed >= color_hold_s:
                    paint_color = color_map[stable][0]
                    color_armed = False
        else:
            color_timer_start = None
            color_target = None
            color_armed = True

        # Overlay final
        out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0.0)

        # ROI
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, "MANO dentro de la caja", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Cursor del color actual (queda muy visual)
        if tip_draw is not None:
            cv2.circle(out, tip_draw, 10, paint_color, -1)
            cv2.circle(out, tip_draw, 10, (255, 255, 255), 2)

        # HUD placement: lado contrario al ROI
        H, W = out.shape[:2]
        roi_on_left = (x2 < W * 0.5)
        hud_w, hud_h = 580, 240
        hud_x = int(0.52 * W) if roi_on_left else 20
        hud_y = 20

        draw_panel(out, hud_x, hud_y, hud_w, hud_h, alpha=0.55)

        frames += 1
        fps = frames / max(time.time() - t0, 1e-6)
        status = "UNLOCKED (PINTAR)" if fsm.unlocked else f"LOCKED {fsm.idx}/{len(pattern)}"

        put_text(out, f"raw: {finger_count} | stable: {stable}", (hud_x + 15, hud_y + 35), 0.75, 2)
        put_text(out, f"Security: {status}", (hud_x + 15, hud_y + 68), 0.75, 2)
        put_text(out, f"maskArea: {area_ratio:.2f} | FPS: {fps:.1f}", (hud_x + 15, hud_y + 101), 0.70, 2)

        draw_password_bar(out, pattern, fsm.idx, fsm.unlocked, hud_x + 15, hud_y + 120)

        # Color + grosor actual
        cv2.rectangle(out, (hud_x + 15, hud_y + 165), (hud_x + 55, hud_y + 205), paint_color, -1)
        cv2.rectangle(out, (hud_x + 15, hud_y + 165), (hud_x + 55, hud_y + 205), (255, 255, 255), 2)
        put_text(out, f"Color | Grosor: {paint_thickness}  (+/-)", (hud_x + 62, hud_y + 195), 0.65, 2)

        put_text(out, "q salir | c limpiar | r reset", (hud_x + 15, hud_y + 230), 0.62, 2)

        if fsm.unlocked:
            if clear_progress is not None:
                draw_progress_bar(out, hud_x + 300, hud_y + 180, 260, 14, clear_progress, "BORRAR (5 dedos 3s)")
            elif color_progress is not None and color_label is not None:
                draw_progress_bar(out, hud_x + 300, hud_y + 180, 260, 14, color_progress, color_label)

        if area_ratio >= 0.60:
            draw_panel(out, hud_x, hud_y + hud_h + 10, hud_w, 55, alpha=0.55)
            put_text(out, "Mask muy blanca -> SUBE Cr_low / BAJA Cb_high", (hud_x + 15, hud_y + hud_h + 45), 0.62, 2)

        # Inset máscara
        draw_inset_mask(out, solid, title="hand solid", pos="bottom_right", size=(260, 190), margin=18)

        cv2.imshow("Hand Password + Paint", out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("r"):
            fsm.reset()
            hist.clear()
            prev_pt = None
            sm_tip = None
            clear_timer_start = None
            clear_armed = True
            color_timer_start = None
            color_target = None
            color_armed = True
            paint_color = (255, 255, 255)
            paint_thickness = 3

        # ====== Grosor con + / - ======
        # '+' suele ser 43, '-' 45. En algunos teclados '+' llega como '=' (61)
        elif key in (ord('+'), ord('=')):
            paint_thickness = min(max_th, paint_thickness + 1)
        elif key in (ord('-'), ord('_')):
            paint_thickness = max(min_th, paint_thickness - 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(cam_index=0, width=1280, height=720)