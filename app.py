import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import json
import io
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Domain Constants
# ─────────────────────────────────────────────
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 1: "Bicycle"}

CLASS_COLORS = {
    "Car":        (0,   200, 255),
    "Motorcycle": (255,  60,   0),   # hot-orange — more visually distinct
    "Bus":        (0,   255, 120),
    "Truck":      (120,   0, 255),
    "Bicycle":    (255, 255,   0),
}

DIRECTION_COLORS = {
    "North": (200, 255, 200),
    "South": (200, 200, 255),
    "East":  (255, 200, 200),
    "West":  (200, 255, 255),
}

# Per-class confidence overrides — motorcycles & bicycles get a lower gate
CLASS_CONF_OVERRIDE = {
    "Motorcycle": 0.18,
    "Bicycle":    0.20,
}


# ─────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────
class VehicleTracker:
    """
    Centroid-based tracker with:
    • Exponential-smoothed speed
    • Per-object direction derived from recent trail
    • IoU-aware distance penalty to reduce ID-switches near overlap
    """

    def __init__(self, max_disappeared=55, max_distance=90):
        self.next_id       = 0
        self.objects       = {}     # oid → (cx, cy)
        self.classes       = {}
        self.trails        = {}
        self.disappeared   = {}
        self.speeds        = {}
        self.directions    = {}
        self.last_seen     = {}
        self.counted_ids   = set()
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    # ── direction helper ──────────────────────
    @staticmethod
    def _get_direction(trail):
        if len(trail) < 6:
            return "Unknown"
        x0, y0 = trail[0]
        x1, y1 = trail[-1]
        dx, dy  = x1 - x0, y1 - y0
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        return "South" if dy > 0 else "North"

    # ── registration ─────────────────────────
    def _register(self, cx, cy, cls_name, frame_idx):
        oid = self.next_id
        self.objects[oid]     = (cx, cy)
        self.classes[oid]     = cls_name
        self.trails[oid]      = [(int(cx), int(cy))]
        self.disappeared[oid] = 0
        self.speeds[oid]      = 0.0
        self.directions[oid]  = "Unknown"
        self.last_seen[oid]   = frame_idx
        self.next_id += 1

    # ── main update ──────────────────────────
    def update(self, detections, fps, scale_mpp, frame_idx, speed_scale=1.0):
        """
        detections: list of (cx, cy, class_name)
        Returns dict: oid → {centroid, class, trail, speed, direction}
        """
        # age existing
        for oid in list(self.disappeared):
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                for d in (self.objects, self.classes, self.trails,
                          self.speeds, self.directions, self.disappeared, self.last_seen):
                    d.pop(oid, None)

        if not detections:
            return self._snapshot()

        input_cents = np.array([(d[0], d[1]) for d in detections], dtype=float)
        input_cls   = [d[2] for d in detections]

        if not self.objects:
            for i in range(len(input_cents)):
                self._register(*input_cents[i], input_cls[i], frame_idx)
            return self._snapshot()

        ids      = list(self.objects.keys())
        existing = np.array(list(self.objects.values()), dtype=float)

        # L2 distance matrix
        D = np.linalg.norm(
            existing[:, None] - input_cents[None, :], axis=2
        )

        rows       = D.min(axis=1).argsort()
        cols       = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue

            oid            = ids[r]
            cx, cy         = float(input_cents[c][0]), float(input_cents[c][1])
            prev_cx, prev_cy = self.objects[oid]
            dist_px        = np.hypot(cx - prev_cx, cy - prev_cy)
            frame_gap      = max(1, frame_idx - self.last_seen.get(oid, frame_idx))
            time_s         = frame_gap / max(fps, 1.0)
            speed_ms       = (dist_px * max(scale_mpp, 0.001)) / max(time_s, 1e-3)
            speed_kh       = min(max(speed_ms * 3.6 * speed_scale, 0.0), 70.0)
            # EMA smoothing + conservative cap to avoid inflated speed spikes
            self.speeds[oid]    = 0.70 * self.speeds.get(oid, speed_kh) + 0.30 * speed_kh
            self.speeds[oid]    = min(max(self.speeds[oid], 0.0), 70.0)
            self.objects[oid]   = (cx, cy)
            self.classes[oid]   = input_cls[c]
            self.disappeared[oid] = 0
            self.trails[oid].append((int(cx), int(cy)))
            if len(self.trails[oid]) > 50:
                self.trails[oid].pop(0)
            self.directions[oid] = self._get_direction(self.trails[oid])
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(ids))) - used_rows:
            self.disappeared[ids[r]] += 1
        for c in set(range(len(input_cents))) - used_cols:
            cx, cy = input_cents[c]
            self._register(cx, cy, input_cls[c], frame_idx)

        return self._snapshot()

    def _snapshot(self):
        return {
            oid: {
                "centroid":  self.objects[oid],
                "class":     self.classes[oid],
                "trail":     self.trails[oid],
                "speed":     self.speeds.get(oid, 0.0),
                "direction": self.directions.get(oid, "Unknown"),
            }
            for oid in self.objects
        }


# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────
class HeatmapOverlay:
    def __init__(self, h, w, decay=0.994):
        self.heat  = np.zeros((h, w), dtype=np.float32)
        self.decay = decay

    def update(self, centroids, radius=28):
        self.heat *= self.decay
        for cx, cy in centroids:
            cv2.circle(self.heat, (int(cx), int(cy)), radius, 0.65, -1)
        self.heat = np.clip(self.heat, 0, 1)

    def render(self, frame, alpha=0.42):
        norm    = (self.heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask    = norm > 15
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
        return overlay


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_trail(frame, trail, color, thickness=2):
    if len(trail) < 2:
        return
    pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
    # fade older segments
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness)


def draw_direction_arrow(frame, trail, color):
    if len(trail) < 4:
        return
    tip  = trail[-1]
    base = trail[-4]
    cv2.arrowedLine(frame, base, tip, color, 2, tipLength=0.4)


def draw_label(frame, text, pos, color, bg=(20, 20, 20)):
    x, y  = pos
    scale = 0.45
    thick = 1
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(frame, (x, y - h - 4), (x + w + 4, y + 2), bg, -1)
    cv2.putText(frame, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PSM Labs · Drone Traffic AI")

st.markdown("""
<style>
  /* dark card feel for expanders */
  [data-testid="stExpander"] { border: 1px solid #2a2a3a; border-radius: 8px; }
  .stButton > button { border-radius: 6px; font-weight: 600; }
  .metric-card { background: #12121f; border: 1px solid #1e1e2e;
                 border-radius: 8px; padding: 10px 14px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 Drone Traffic Analysis — PSM Labs")
st.markdown("*Multi-class detection · Speed estimation · Direction tracking · Density heatmap*")
st.markdown("---")

left_col, right_col = st.columns([1, 1.15], gap="large")

# ─────────────────────────────────────────────
# LEFT PANEL — Controls & Export
# ─────────────────────────────────────────────
with left_col:
    st.subheader("📥 Input & Configuration")
    uploaded_file = st.file_uploader(
        "Upload traffic video", type=["mp4", "avi", "mov", "mkv"]
    )

    # ── Model ──────────────────────────────────
    with st.expander("🧠 Model Settings", expanded=True):
        model_choice = st.selectbox(
            "YOLOv8 Variant",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=1,
            help="'s' or 'm' recommended for motorcycle accuracy. 'n' is fastest.",
        )
        conf_threshold = st.slider("Base Confidence Threshold", 0.10, 0.90, 0.22, 0.01)
        iou_threshold  = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.40, 0.05)
        multi_scale    = st.checkbox(
            "Multi-scale Inference (better small-object recall)",
            value=True,
            help="Runs inference at two scales and merges; adds ~30% compute but significantly improves motorcycle & bicycle detection.",
        )

    # ── Analysis ───────────────────────────────
    with st.expander("📐 Analysis Parameters", expanded=True):
        line_position = st.slider("Counting Line Position (fraction of height)", 0.1, 0.9, 0.50, 0.05)
        scale_mpp     = st.number_input("Scale Calibration (meters / pixel)", value=0.05, step=0.005, format="%.3f")
        speed_scale   = st.number_input("Speed Scale Factor", value=0.45, min_value=0.1, max_value=1.0, step=0.05,
                                        help="Lower values make speed estimates more conservative.")
        enable_heatmap = st.checkbox("Density Heatmap Overlay", value=True)
        show_trails    = st.checkbox("Vehicle Trail Visualization", value=True)
        show_speed     = st.checkbox("Show Speed Labels", value=True)

    # ── Export Options ─────────────────────────
    with st.expander("📤 Data Export Options", expanded=True):
        st.markdown("**Select data to extract after processing:**")
        export_counts     = st.checkbox("Vehicle Count Summary", value=True)
        export_speed_log  = st.checkbox("Full Speed & Direction Log", value=True)
        export_direction  = st.checkbox("Direction Breakdown per Class", value=True)
        export_peak       = st.checkbox("Peak Traffic Interval (per 30 s)", value=True)
        export_incidents  = st.checkbox("Speeding Incidents (threshold-based)", value=False)
        speed_limit_kmh   = st.number_input(
            "Speed Limit for Incident Flag (km/h)", value=60.0, step=5.0,
            disabled=not export_incidents,
        )
        export_format = st.radio("Export Format", ["CSV", "JSON"], horizontal=True)

    run_analysis = st.button(
        "🚀 Start Analysis Pipeline", use_container_width=True, type="primary"
    )

    # Export download area — filled after processing
    export_placeholder = st.empty()

# ─────────────────────────────────────────────
# RIGHT PANEL — Live Feed + Metrics
# ─────────────────────────────────────────────
with right_col:
    st.subheader("📡 Live Feed & Analytics")
    with st.container(border=True):
        video_placeholder  = st.empty()
        metric_placeholder = st.empty()

    tab_log, tab_dir, tab_speed = st.tabs(["📋 Event Log", "🧭 Direction Breakdown", "⚡ Speed Histogram"])
    with tab_log:
        log_placeholder = st.empty()
    with tab_dir:
        dir_placeholder = st.empty()
    with tab_speed:
        spd_placeholder = st.empty()


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
if run_analysis and uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    LINE_Y = int(H * line_position)

    # ── Load model ─────────────────────────────
    with st.spinner(f"Loading {model_choice}..."):
        model = YOLO(model_choice)

    tracker  = VehicleTracker(max_disappeared=55, max_distance=90)
    heatmap  = HeatmapOverlay(H, W)

    total_counts    = defaultdict(int)    # class → count
    direction_counts = defaultdict(lambda: defaultdict(int))  # class → dir → count
    speed_log       = []
    interval_counts = defaultdict(int)   # 30-s bucket → total
    frame_idx       = 0
    INTERVAL_FRAMES = int(FPS * 30)      # frames per 30-second bucket

    progress_bar = st.progress(0.0, text="Processing frames…")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        progress_bar.progress(
            min(frame_idx / total_frames, 1.0),
            text=f"Frame {frame_idx} / {total_frames}"
        )

        # ── Inference ──────────────────────────
        # Primary pass
        results = model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            augment=multi_scale,
            verbose=False,
        )[0]

        # Secondary small-scale pass for motorcycles / bicycles if multi_scale is on
        extra_boxes = []
        if multi_scale:
            results2 = model(
                frame,
                conf=CLASS_CONF_OVERRIDE["Motorcycle"],   # lower gate
                iou=iou_threshold,
                imgsz=320,                                # smaller → faster
                augment=False,
                verbose=False,
            )[0]
            for box in results2.boxes:
                cls_id = int(box.cls[0])
                if cls_id in (3, 1):   # Motorcycle, Bicycle only
                    extra_boxes.append(box)

        all_boxes = list(results.boxes) + extra_boxes

        detections = []
        seen_positions = []   # simple dedup for secondary pass

        for box in all_boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            cls_name = VEHICLE_CLASSES[cls_id]
            # per-class confidence gate
            min_conf = CLASS_CONF_OVERRIDE.get(cls_name, conf_threshold)
            if conf_val < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # deduplicate secondary-pass detections within 25 px
            too_close = any(
                abs(cx - px) < 25 and abs(cy - py) < 25
                for px, py in seen_positions
            )
            if too_close:
                continue
            seen_positions.append((cx, cy))

            detections.append((cx, cy, cls_name))
            color = CLASS_COLORS[cls_name]
            thick = 3 if cls_name == "Motorcycle" else 2   # thicker box for bikes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            draw_label(frame, f"{cls_name} {conf_val:.0%}", (x1, y1), color)

        # ── Track ──────────────────────────────
        tracked = tracker.update(detections, FPS, scale_mpp, frame_idx, speed_scale)

        # ── Count + log at line ─────────────────
        for oid, info in tracked.items():
            cx, cy   = info["centroid"]
            cls_name = info["class"]
            direction = info["direction"]

            if oid not in tracker.counted_ids and abs(cy - LINE_Y) < 20:
                tracker.counted_ids.add(oid)
                total_counts[cls_name] += 1
                direction_counts[cls_name][direction] += 1
                bucket = frame_idx // INTERVAL_FRAMES
                interval_counts[bucket] += 1
                speed_log.append({
                    "Timestamp (s)":  round(frame_idx / FPS, 2),
                    "Vehicle ID":     oid,
                    "Class":          cls_name,
                    "Direction":      direction,
                    "Speed (km/h)":   round(info["speed"], 1),
                    "Frame":          frame_idx,
                })

            # trails & arrows
            if show_trails:
                draw_trail(frame, info["trail"], CLASS_COLORS[cls_name])
                draw_direction_arrow(frame, info["trail"], DIRECTION_COLORS.get(direction, (255, 255, 255)))

            if show_speed and info["speed"] > 1.0:
                cx_i, cy_i = int(cx), int(cy)
                draw_label(
                    frame,
                    f"{info['speed']:.0f} km/h",
                    (cx_i + 5, cy_i - 8),
                    CLASS_COLORS[cls_name],
                )

        # ── Heatmap ────────────────────────────
        if enable_heatmap:
            heatmap.update([info["centroid"] for info in tracked.values()])
            frame = heatmap.render(frame)

        # ── Counting line ──────────────────────
        cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 255, 255), 2)
        draw_label(frame, "COUNT LINE", (8, LINE_Y - 4), (0, 255, 255))

        # ── Display ────────────────────────────
        video_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        # ── Metrics ────────────────────────────
        with metric_placeholder.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total",       sum(total_counts.values()))
            c2.metric("Cars",        total_counts.get("Car", 0))
            c3.metric("Motorcycles", total_counts.get("Motorcycle", 0))
            c4.metric("Trucks",      total_counts.get("Truck", 0))
            c5.metric("Buses",       total_counts.get("Bus", 0))

        # ── Live tabs ──────────────────────────
        if speed_log:
            df_log = pd.DataFrame(speed_log)
            log_placeholder.dataframe(df_log.tail(10), use_container_width=True)

            # Direction breakdown
            dir_rows = []
            for cls, dirs in direction_counts.items():
                for d, cnt in dirs.items():
                    dir_rows.append({"Class": cls, "Direction": d, "Count": cnt})
            if dir_rows:
                dir_placeholder.dataframe(
                    pd.DataFrame(dir_rows).sort_values("Count", ascending=False),
                    use_container_width=True,
                )

            # Speed histogram (bar chart)
            spd_df = df_log[["Class", "Speed (km/h)"]].copy()
            spd_hist = spd_df.groupby("Class")["Speed (km/h)"].mean().round(1).reset_index()
            spd_hist.columns = ["Class", "Avg Speed (km/h)"]
            spd_placeholder.bar_chart(
                spd_hist.set_index("Class"), use_container_width=True
            )

    cap.release()
    progress_bar.progress(1.0, text="✅ Processing complete")
    st.success("🎉 Video processing complete!")

    # ─────────────────────────────────────────
    # Export Assembly
    # ─────────────────────────────────────────
    df_log = pd.DataFrame(speed_log) if speed_log else pd.DataFrame()

    export_data = {}

    if export_counts:
        export_data["vehicle_count_summary"] = dict(total_counts)

    if export_speed_log and not df_log.empty:
        export_data["speed_direction_log"] = df_log.to_dict(orient="records")

    if export_direction:
        dir_export = {cls: dict(dirs) for cls, dirs in direction_counts.items()}
        export_data["direction_breakdown"] = dir_export

    if export_peak and interval_counts:
        peak_rows = [
            {"Interval (30s block)": k, "Vehicle Count": v}
            for k, v in sorted(interval_counts.items())
        ]
        export_data["peak_traffic_intervals"] = peak_rows

    if export_incidents and not df_log.empty:
        incidents = df_log[df_log["Speed (km/h)"] > speed_limit_kmh].to_dict(orient="records")
        export_data["speeding_incidents"] = incidents

    # ── Download button ───────────────────────
    with export_placeholder.container():
        st.markdown("### 📥 Download Extracted Data")
        if not export_data:
            st.info("No export options selected.")
        else:
            if export_format == "JSON":
                raw = json.dumps(export_data, indent=2)
                st.download_button(
                    "⬇️ Download JSON Report",
                    data=raw,
                    file_name="traffic_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:  # CSV — one sheet per section, zipped
                import zipfile, os

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # counts
                    if "vehicle_count_summary" in export_data:
                        cnt_df = pd.DataFrame(
                            list(export_data["vehicle_count_summary"].items()),
                            columns=["Class", "Count"],
                        )
                        zf.writestr("vehicle_counts.csv", cnt_df.to_csv(index=False))

                    if "speed_direction_log" in export_data:
                        zf.writestr(
                            "speed_direction_log.csv",
                            df_log.to_csv(index=False),
                        )

                    if "direction_breakdown" in export_data:
                        dir_rows2 = [
                            {"Class": c, "Direction": d, "Count": v}
                            for c, dirs in direction_counts.items()
                            for d, v in dirs.items()
                        ]
                        zf.writestr(
                            "direction_breakdown.csv",
                            pd.DataFrame(dir_rows2).to_csv(index=False),
                        )

                    if "peak_traffic_intervals" in export_data:
                        pk_df = pd.DataFrame(export_data["peak_traffic_intervals"])
                        zf.writestr("peak_traffic.csv", pk_df.to_csv(index=False))

                    if "speeding_incidents" in export_data:
                        inc_df = pd.DataFrame(export_data["speeding_incidents"])
                        if not inc_df.empty:
                            zf.writestr("speeding_incidents.csv", inc_df.to_csv(index=False))

                zip_buf.seek(0)
                st.download_button(
                    "⬇️ Download CSV Package (.zip)",
                    data=zip_buf,
                    file_name="traffic_analysis_csvs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

elif run_analysis and uploaded_file is None:
    st.error("⚠️ Please upload a video file before starting the analysis.")