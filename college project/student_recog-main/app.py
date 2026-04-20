import json
import os
import sys
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "model"
STUDENTS_FILE = DATA_DIR / "students.json"
TEACHERS_FILE = DATA_DIR / "teachers.json"
TRAINER_FILE = MODEL_DIR / "trainer.yml"
TIMETABLE_FILE = DATA_DIR / "timetable.json"
ALERTS_DIR = DATA_DIR / "alerts"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_IMAGE_SIZE = (100, 100)
DETECTION_RESIZE_SCALE = 0.6
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 5
CAPTURE_SAMPLE_INTERVAL_SECONDS = 0.18
CAPTURE_DUPLICATE_THRESHOLD = 10.0
RECOGNITION_FRAME_SKIP = 1
RECOGNITION_STABLE_FRAMES = 3
DEFAULT_CAPTURE_SAMPLES = 25
DEFAULT_CONFIDENCE_THRESHOLD = 95.0
MAX_MONITORED_FACES = 25
IGNORED_ACTIVITIES = {
    "break",
    "lunch",
    "sports",
    "sport",
    "yoga",
    "net lab",
    "library",
    "club activity",
}
ALERT_COOLDOWN_SECONDS = 600
JSON_CACHE = {}


def default_teachers() -> dict:
    return {
        "default_alert_teacher_ids": ["teacher_1", "teacher_2"],
        "teachers": {
            "teacher_1": {
                "name": "Teacher 1",
                "phone": "9897057701",
            },
            "teacher_2": {
                "name": "Teacher 2",
                "phone": "8791301224",
            },
        },
    }


def default_timetable() -> dict:
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    template = [
        {"start": "09:00", "end": "09:50", "activity": "Math"},
        {"start": "09:50", "end": "10:40", "activity": "Science"},
        {"start": "10:40", "end": "11:00", "activity": "Break"},
        {"start": "11:00", "end": "11:50", "activity": "English"},
        {"start": "11:50", "end": "12:40", "activity": "Social Studies"},
        {"start": "12:40", "end": "13:20", "activity": "Lunch"},
        {"start": "13:20", "end": "14:10", "activity": "Computer"},
        {"start": "14:10", "end": "15:00", "activity": "Library"},
    ]
    return {
        "default_section": "A",
        "timezone_note": "Times are interpreted using the computer's local clock.",
        "sections": {
            "A": {
                "teacher_ids": ["teacher_1", "teacher_2"],
                "days": {
                    weekday: ([*template] if weekday != "sunday" else [])
                    for weekday in weekdays
                }
            }
        },
    }


def read_json_file(file_path: Path, default):
    if not file_path.exists():
        return default

    cache_key = str(file_path)
    try:
        mtime_ns = file_path.stat().st_mtime_ns
    except OSError:
        return default

    cached_entry = JSON_CACHE.get(cache_key)
    if cached_entry and cached_entry["mtime_ns"] == mtime_ns:
        return cached_entry["data"]

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        JSON_CACHE[cache_key] = {"mtime_ns": mtime_ns, "data": data}
        return data
    except json.JSONDecodeError:
        return default


def write_json_file(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    JSON_CACHE[str(file_path)] = {
        "mtime_ns": file_path.stat().st_mtime_ns,
        "data": data,
    }


def ensure_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    if not STUDENTS_FILE.exists():
        STUDENTS_FILE.write_text("{}", encoding="utf-8")
    if not TEACHERS_FILE.exists():
        write_json_file(TEACHERS_FILE, default_teachers())
    if not TIMETABLE_FILE.exists():
        write_json_file(TIMETABLE_FILE, default_timetable())

def require_lbph():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH recognizer is unavailable.\n"
            "Install it with: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def load_students() -> dict:
    ensure_directories()
    return read_json_file(STUDENTS_FILE, {})


def save_students(students: dict) -> None:
    write_json_file(STUDENTS_FILE, students)


def load_timetable() -> dict:
    ensure_directories()
    return read_json_file(TIMETABLE_FILE, default_timetable())


def load_teachers() -> dict:
    ensure_directories()
    return read_json_file(TEACHERS_FILE, default_teachers())


def clean_student_value(value) -> str:
    return str(value or "").strip()


def get_sections() -> dict:
    timetable = load_timetable()
    return timetable.get("sections", {})


def get_default_section() -> str:
    timetable = load_timetable()
    default_section = timetable.get("default_section", "").strip()
    sections = get_sections()
    if default_section and default_section in sections:
        return default_section
    if sections:
        return next(iter(sections))
    return ""


def resolve_section(section_name: str | None = None) -> str:
    sections = get_sections()
    if not sections:
        return ""

    if section_name:
        normalized = section_name.strip().upper()
        for existing in sections:
            if existing.upper() == normalized:
                return existing

    return get_default_section()


def match_section_name(section_name: str | None = None) -> str:
    sections = get_sections()
    if not sections or not section_name:
        return ""

    normalized = section_name.strip().upper()
    for existing in sections:
        if existing.upper() == normalized:
            return existing
    return ""


def build_timetable_candidates(student: dict) -> list[str]:
    candidates = []
    raw_values = [
        student.get("section"),
        student.get("class_name"),
    ]

    course = clean_student_value(student.get("course")).upper().replace(" ", "")
    year = clean_student_value(student.get("year")).upper().replace(" ", "")
    class_name = clean_student_value(student.get("class_name")).upper().replace(" ", "")

    if course and year:
        raw_values.extend(
            [
                f"{course}{year}",
                f"{course}-{year}",
                f"{course} {year}",
            ]
        )

    if class_name and course and year:
        raw_values.append(f"{course}{year}{class_name}")

    for value in raw_values:
        cleaned = clean_student_value(value)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    return candidates


def resolve_student_section(student: dict) -> str:
    for candidate in build_timetable_candidates(student):
        resolved = match_section_name(candidate)
        if resolved:
            return resolved
    return ""


def get_face_detector():
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def prepare_face_image(face_region) -> np.ndarray:
    face_region = cv2.resize(face_region, FACE_IMAGE_SIZE)
    return cv2.equalizeHist(face_region)


def prepare_color_face_image(face_region) -> np.ndarray:
    return cv2.resize(face_region, FACE_IMAGE_SIZE)


def detect_faces(detector, gray_frame):
    small_frame = cv2.resize(
        gray_frame,
        None,
        fx=DETECTION_RESIZE_SCALE,
        fy=DETECTION_RESIZE_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    small_frame = cv2.equalizeHist(small_frame)
    detected_faces = detector.detectMultiScale(
        small_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40),
    )

    scale_back = 1.0 / DETECTION_RESIZE_SCALE
    faces = []
    for (x, y, w, h) in detected_faces:
        faces.append(
            (
                int(x * scale_back),
                int(y * scale_back),
                int(w * scale_back),
                int(h * scale_back),
            )
        )
    faces.sort(key=lambda face: face[2] * face[3], reverse=True)
    return faces[:MAX_MONITORED_FACES]


def make_face_tracking_key(student_id: str, confidence: float, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    grid_x = x // 50
    grid_y = y // 50
    if student_id and confidence >= 0:
        return f"student:{student_id}:{grid_x}:{grid_y}"
    return f"unknown:{grid_x}:{grid_y}:{w // 20}:{h // 20}"


def configure_camera(camera) -> None:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera(index: int = 0):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    configure_camera(camera)
    for _ in range(CAMERA_WARMUP_FRAMES):
        camera.read()
    return camera


def should_close_window(window_name: str) -> bool:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True
    except cv2.error:
        return False

    key = cv2.waitKey(10) & 0xFF
    return key in (ord("q"), ord("Q"), 27)


def face_difference_score(first_face: np.ndarray | None, second_face: np.ndarray) -> float:
    if first_face is None:
        return float("inf")
    return float(np.mean(cv2.absdiff(first_face, second_face)))


def compute_training_signature(label_map: dict) -> str:
    hasher = sha256()

    for label, student_id in sorted(label_map.items()):
        hasher.update(f"{label}:{student_id}|".encode("utf-8"))
        student_dir = DATASET_DIR / student_id
        for image_path in sorted(student_dir.glob("*.jpg")):
            stat = image_path.stat()
            hasher.update(
                f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode("utf-8")
            )

    return hasher.hexdigest()


def save_training_metadata(metadata: dict) -> None:
    write_json_file(MODEL_DIR / "training_meta.json", metadata)


def load_training_metadata() -> dict:
    return read_json_file(MODEL_DIR / "training_meta.json", {})


def should_skip_training(label_map: dict) -> bool:
    if not TRAINER_FILE.exists():
        return False

    metadata = load_training_metadata()
    current_signature = compute_training_signature(label_map)
    return metadata.get("signature") == current_signature


def capture_student_faces(
    student_id: str,
    name: str,
    guardian_phone: str,
    guardian_name: str = "",
    course: str = "",
    class_name: str = "",
    year: str = "",
    section: str = "",
    samples: int = DEFAULT_CAPTURE_SAMPLES,
) -> None:
    window_name = "Capture Student Faces"
    detector = get_face_detector()
    students = load_students()
    existing_student = students.get(student_id, {})
    students[student_id] = {
        "name": name,
        "guardian_phone": guardian_phone,
        "guardian_name": guardian_name or existing_student.get("guardian_name", ""),
        "course": course or existing_student.get("course", ""),
        "class_name": class_name or existing_student.get("class_name", ""),
        "year": year or existing_student.get("year", ""),
        "section": section or existing_student.get("section", ""),
    }
    save_students(students)

    student_dir = DATASET_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    camera = open_camera()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\nCapturing face samples for {name} ({student_id})")
    print("Press 'q', 'Q', Esc, or close the window to stop early.\n")

    saved = 0
    last_saved_at = 0.0
    last_saved_face = None

    while saved < samples:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        for (x, y, w, h) in faces:
            color_face_region = frame[y : y + h, x : x + w]
            face_region = gray[y : y + h, x : x + w]
            face_region = prepare_face_image(face_region)
            now = time.time()
            difference_score = face_difference_score(last_saved_face, face_region)

            if now - last_saved_at < CAPTURE_SAMPLE_INTERVAL_SECONDS:
                continue
            if difference_score < CAPTURE_DUPLICATE_THRESHOLD:
                continue

            file_path = student_dir / f"{saved + 1:03d}.jpg"
            cv2.imwrite(str(file_path), prepare_color_face_image(color_face_region))
            saved += 1
            last_saved_at = now
            last_saved_face = face_region

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {saved}/{samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 255, 50),
                2,
            )
            cv2.putText(
                frame,
                f"Change: {difference_score:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 220),
                2,
            )
            break

        cv2.imshow(window_name, frame)
        if should_close_window(window_name):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} face image(s) to {student_dir}")


def load_training_data():
    students = load_students()
    faces = []
    labels = []
    label_map = {}
    next_label = 1

    for student_id in sorted(students.keys()):
        student_dir = DATASET_DIR / student_id
        if not student_dir.exists():
            continue

        image_paths = sorted(student_dir.glob("*.jpg"))
        if not image_paths:
            continue

        label_map[next_label] = student_id
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(prepare_face_image(image))
            labels.append(next_label)

        next_label += 1

    return faces, labels, label_map


def save_label_map(label_map: dict) -> None:
    label_map_path = MODEL_DIR / "labels.json"
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def load_label_map() -> dict:
    label_map_path = MODEL_DIR / "labels.json"
    if not label_map_path.exists():
        return {}
    return read_json_file(label_map_path, {})


def parse_minutes(value: str) -> int:
    hour_text, minute_text = value.split(":")
    return int(hour_text) * 60 + int(minute_text)


def get_day_slots(section_name: str | None = None, day_name: str | None = None) -> list:
    schedule = load_timetable()
    resolved_section = resolve_section(section_name)
    day_name = day_name or datetime.now().strftime("%A").lower()

    if resolved_section:
        section_days = schedule.get("sections", {}).get(resolved_section, {}).get("days", {})
        if day_name in section_days:
            return section_days.get(day_name, [])

    return schedule.get("days", {}).get(day_name, schedule.get(day_name, []))


def get_current_schedule_slot(
    now: datetime | None = None, section_name: str | None = None
) -> dict | None:
    now = now or datetime.now()
    day_name = now.strftime("%A").lower()
    day_slots = get_day_slots(section_name, day_name)
    current_minutes = now.hour * 60 + now.minute

    for slot in day_slots:
        try:
            start = parse_minutes(slot["start"])
            end = parse_minutes(slot["end"])
        except (KeyError, ValueError):
            continue
        if start <= current_minutes < end:
            return slot
    return None


def is_ignored_activity(activity: str) -> bool:
    normalized = activity.strip().lower()
    return any(ignored in normalized for ignored in IGNORED_ACTIVITIES)


def get_monitoring_decision(
    now: datetime | None = None, section_name: str | None = None
) -> tuple[bool, str, dict | None]:
    resolved_section = resolve_section(section_name)
    slot = get_current_schedule_slot(now, resolved_section)
    if not slot:
        if resolved_section:
            return (
                False,
                f"No active timetable slot right now for section {resolved_section}.",
                None,
            )
        return False, "No timetable slot is active right now.", None

    activity = slot.get("activity", "Unknown")
    if is_ignored_activity(activity):
        if resolved_section:
            return False, f"{resolved_section}: monitoring paused for {activity}.", slot
        return False, f"Monitoring paused for {activity}.", slot

    if resolved_section:
        return True, f"{resolved_section}: monitoring during {activity}.", slot
    return True, f"Monitoring during {activity}.", slot


def normalize_activity_name(activity: str) -> str:
    return activity.strip().lower()


def split_phone_numbers(value) -> list[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value or "").split(",")
    numbers = []
    for part in parts:
        phone = str(part).strip()
        if phone:
            numbers.append(phone)
    return numbers


def make_contact(name: str, phone: str, role: str) -> dict | None:
    cleaned_name = str(name or "").strip()
    cleaned_phone = str(phone or "").strip()
    if not cleaned_phone:
        return None
    return {
        "name": cleaned_name or role.title(),
        "phone": cleaned_phone,
        "role": role,
    }


def dedupe_contacts(contacts: list[dict]) -> list[dict]:
    unique_contacts = []
    seen = set()
    for contact in contacts:
        phone = str(contact.get("phone", "")).strip()
        if not phone or phone in seen:
            continue
        seen.add(phone)
        unique_contacts.append(contact)
    return unique_contacts


def get_parent_recipients(student: dict) -> list[dict]:
    contacts = []

    if isinstance(student.get("parent_contacts"), list):
        for index, parent_contact in enumerate(student["parent_contacts"], start=1):
            if not isinstance(parent_contact, dict):
                continue
            contact = make_contact(
                parent_contact.get("name", f"Parent {index}"),
                parent_contact.get("phone", ""),
                "parent",
            )
            if contact:
                contacts.append(contact)

    guardian_name = student.get("guardian_name", "Parent/Guardian")
    for phone in split_phone_numbers(student.get("guardian_phone", "")):
        contact = make_contact(guardian_name, phone, "parent")
        if contact:
            contacts.append(contact)

    return dedupe_contacts(contacts)


def teacher_contacts_from_ids(teacher_ids: list[str], teachers_data: dict) -> list[dict]:
    contacts = []
    teachers = teachers_data.get("teachers", {})
    for teacher_id in teacher_ids:
        teacher = teachers.get(teacher_id, {})
        contact = make_contact(teacher.get("name", teacher_id), teacher.get("phone", ""), "teacher")
        if contact:
            contact["teacher_id"] = teacher_id
            contacts.append(contact)
    return contacts


def get_teacher_recipients(section_name: str, slot: dict | None) -> list[dict]:
    timetable = load_timetable()
    teachers_data = load_teachers()
    section = timetable.get("sections", {}).get(section_name, {})
    contacts = []

    if slot:
        if isinstance(slot.get("teacher_contacts"), list):
            for teacher_contact in slot["teacher_contacts"]:
                if not isinstance(teacher_contact, dict):
                    continue
                contact = make_contact(
                    teacher_contact.get("name", ""),
                    teacher_contact.get("phone", ""),
                    "teacher",
                )
                if contact:
                    contacts.append(contact)

        teacher_name = slot.get("teacher_name", "")
        teacher_phone = slot.get("teacher_phone", "")
        contact = make_contact(teacher_name, teacher_phone, "teacher")
        if contact:
            contacts.append(contact)

        if isinstance(slot.get("teacher_ids"), list):
            contacts.extend(teacher_contacts_from_ids(slot["teacher_ids"], teachers_data))

    activity = normalize_activity_name(slot.get("activity", "")) if slot else ""
    teachers_by_activity = section.get("teachers_by_activity", {})
    if activity and isinstance(teachers_by_activity, dict):
        activity_teachers = teachers_by_activity.get(activity, {})
        if isinstance(activity_teachers, dict):
            contact = make_contact(
                activity_teachers.get("name", ""),
                activity_teachers.get("phone", ""),
                "teacher",
            )
            if contact:
                contacts.append(contact)
            if isinstance(activity_teachers.get("teacher_ids"), list):
                contacts.extend(
                    teacher_contacts_from_ids(activity_teachers["teacher_ids"], teachers_data)
                )
        elif isinstance(activity_teachers, list):
            for teacher_contact in activity_teachers:
                if not isinstance(teacher_contact, dict):
                    continue
                contact = make_contact(
                    teacher_contact.get("name", ""),
                    teacher_contact.get("phone", ""),
                    "teacher",
                )
                if contact:
                    contacts.append(contact)

    subject_teacher_ids = timetable.get("subject_teacher_ids", {})
    if activity and isinstance(subject_teacher_ids, dict):
        teacher_ids = subject_teacher_ids.get(activity, [])
        if isinstance(teacher_ids, str):
            teacher_ids = [teacher_ids]
        if isinstance(teacher_ids, list):
            contacts.extend(teacher_contacts_from_ids(teacher_ids, teachers_data))

    if isinstance(section.get("teacher_contacts"), list):
        for teacher_contact in section["teacher_contacts"]:
            if not isinstance(teacher_contact, dict):
                continue
            contact = make_contact(
                teacher_contact.get("name", ""),
                teacher_contact.get("phone", ""),
                "teacher",
            )
            if contact:
                contacts.append(contact)

    if isinstance(section.get("teacher_ids"), list):
        contacts.extend(teacher_contacts_from_ids(section["teacher_ids"], teachers_data))

    if not contacts and isinstance(teachers_data.get("default_alert_teacher_ids"), list):
        contacts.extend(
            teacher_contacts_from_ids(teachers_data["default_alert_teacher_ids"], teachers_data)
        )

    return dedupe_contacts(contacts)


def get_student_display_group(student: dict, resolved_section: str = "") -> str:
    return (
        resolved_section
        or clean_student_value(student.get("section"))
        or "Unassigned"
    )


def build_teacher_alert_message(
    student_id: str,
    student: dict,
    student_group: str,
    activity: str,
    timestamp: str,
) -> str:
    return (
        f"THIS STUDENT IS NOT PRESENT IN YOUR CLASS: "
        f"{student.get('name', 'Unknown Student')} (ID: {student_id}) | "
        f"Class: {student_group} | Course: {student.get('course', '')} | "
        f"Year: {student.get('year', '')} | Subject: {activity} | Time: {timestamp}"
    )


def build_parent_alert_message(
    student_id: str,
    student: dict,
    student_group: str,
    activity: str,
    timestamp: str,
) -> str:
    return (
        f"Dear Parent/Guardian, your ward {student.get('name', 'Unknown Student')} "
        f"(ID: {student_id}) from {student_group} may be bunking {activity}. "
        f"The student was detected outside the scheduled class at {timestamp}. "
        f"Please verify the student's attendance."
    )


def build_alert_payload(
    student_id: str,
    student: dict,
    section_name: str,
    slot: dict | None,
    image_path: Path,
) -> dict:
    activity = slot.get("activity", "class period") if slot else "class period"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    student_group = get_student_display_group(student, section_name)
    parent_contacts = get_parent_recipients(student)
    teacher_contacts = get_teacher_recipients(section_name, slot)
    recipients = dedupe_contacts(parent_contacts + teacher_contacts)

    recipients_text = ", ".join(
        f"{contact['name']} ({contact['phone']})" for contact in recipients
    ) or "No contacts configured"
    teacher_message = build_teacher_alert_message(
        student_id,
        student,
        student_group,
        activity,
        timestamp,
    )
    parent_message = build_parent_alert_message(
        student_id,
        student,
        student_group,
        activity,
        timestamp,
    )
    message = (
        f"{teacher_message} Notify: {recipients_text}. "
        f"Parent copy: {parent_message}"
    )

    payload = {
        "student_id": student_id,
        "student_name": student.get("name", ""),
        "section": student_group,
        "class_name": student.get("class_name", ""),
        "course": student.get("course", ""),
        "year": student.get("year", ""),
        "guardian_phone": student.get("guardian_phone", ""),
        "parent_contacts": parent_contacts,
        "parent_recipients": parent_contacts,
        "activity": activity,
        "timestamp": timestamp,
        "message": message,
        "teacher_message": teacher_message,
        "parent_message": parent_message,
        "image_path": str(image_path),
        "image_url": "",
        "teacher_contacts": teacher_contacts,
        "teacher_recipients": teacher_contacts,
        "recipients": recipients,
        "alert_type": "bunk_detection",
    }

    image_base_url = os.getenv("ALERT_IMAGE_BASE_URL", "").rstrip("/")
    if image_base_url:
        payload["image_url"] = f"{image_base_url}/{image_path.name}"

    return payload


def save_alert_snapshot(frame, student_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_path = ALERTS_DIR / f"{student_id}_{timestamp}.jpg"
    cv2.imwrite(str(alert_path), frame)
    return alert_path


def send_alert(
    student_id: str,
    student: dict,
    section_name: str,
    slot: dict | None,
    image_path: Path,
) -> bool:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False

    payload = build_alert_payload(student_id, student, section_name, slot, image_path)

    req = request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = os.getenv("ALERT_WEBHOOK_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with request.urlopen(req, timeout=10) as response:
            return 200 <= response.status < 300
    except (error.URLError, error.HTTPError):
        return False


def train_model() -> None:
    recognizer = require_lbph()
    faces, labels, label_map = load_training_data()

    if not faces:
        raise RuntimeError(
            "No training data found. Register a student and capture face samples first."
        )

    if should_skip_training(label_map):
        print("Training skipped because the dataset has not changed.")
        print(f"Existing trained model is already up to date at {TRAINER_FILE}")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(str(TRAINER_FILE))
    save_label_map(label_map)
    save_training_metadata(
        {
            "signature": compute_training_signature(label_map),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "image_count": len(faces),
            "student_count": len(label_map),
        }
    )

    print(f"Model trained successfully with {len(faces)} images.")
    print(f"Saved trained model to {TRAINER_FILE}")


def recognize_faces(
    section_name: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    if not TRAINER_FILE.exists():
        raise RuntimeError("Trained model not found. Please train the model first.")
    if not get_sections():
        raise RuntimeError("No timetable sections found. Please check timetable.json.")

    recognizer = require_lbph()
    recognizer.read(str(TRAINER_FILE))
    label_map = load_label_map()
    students = load_students()
    detector = get_face_detector()

    camera = open_camera()
    window_name = "Student Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nRunning live recognition for all registered students.")
    print("Press 'q', 'Q', Esc, or close the window to quit.\n")
    print(
        f"Loaded {len(label_map)} trained student(s). "
        f"Recognition threshold: {confidence_threshold:.1f}"
    )

    last_alert_times = {}
    frame_index = 0
    recognized_faces = []
    recognition_history = {}

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame_index += 1
        cv2.putText(
            frame,
            "Scanning students and checking their own timetable...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if frame_index % (RECOGNITION_FRAME_SKIP + 1) == 1 or not recognized_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, gray)
            new_recognized_faces = []
            next_recognition_history = {}

            cv2.putText(
                frame,
                f"Faces in frame: {len(faces)} / {MAX_MONITORED_FACES}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 220, 0),
                2,
            )

            for (x, y, w, h) in faces:
                face_region = gray[y : y + h, x : x + w]
                face_region = prepare_face_image(face_region)
                label, confidence = recognizer.predict(face_region)

                student_id = label_map.get(str(label)) or label_map.get(label)
                student = students.get(student_id, {})
                student_section = resolve_student_section(student)

                tracked_student_id = student_id if confidence < confidence_threshold and student else ""
                identity_key = make_face_tracking_key(
                    tracked_student_id,
                    confidence,
                    (x, y, w, h),
                )
                history_count = recognition_history.get(identity_key, 0) + 1
                next_recognition_history[identity_key] = history_count
                is_stable = history_count >= RECOGNITION_STABLE_FRAMES

                if confidence < confidence_threshold and student and is_stable:
                    if not student_section:
                        text = (
                            f"{student['name']} | ID: {student_id} | "
                            "No matching timetable section"
                        )
                        color = (0, 165, 255)
                    else:
                        monitoring_enabled, _, active_slot = get_monitoring_decision(
                            section_name=student_section
                        )
                        current_activity = (
                            active_slot.get("activity", "No active class")
                            if active_slot
                            else "No active class"
                        )
                        text = (
                            f"{student['name']} | ID: {student_id} | "
                            f"Section: {student_section} | {current_activity}"
                        )
                        color = (50, 220, 50)
                    if student_section and monitoring_enabled and active_slot:
                        current_time = datetime.now().timestamp()
                        last_alert_time = last_alert_times.get(student_id, 0.0)
                        if current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                            image_path = save_alert_snapshot(frame, student_id)
                            alert_payload = build_alert_payload(
                                student_id,
                                student,
                                student_section,
                                active_slot,
                                image_path,
                            )
                            alert_sent = send_alert(
                                student_id,
                                student,
                                student_section,
                                active_slot,
                                image_path,
                            )
                            last_alert_times[student_id] = current_time
                            recipient_summary = ", ".join(
                                f"{contact['name']} ({contact['phone']})"
                                for contact in alert_payload.get("recipients", [])
                            ) or "no configured recipients"
                            print(
                                f"Alert {'sent' if alert_sent else 'saved locally'} for "
                                f"{student['name']} ({student_id}) at {image_path}. "
                                f"Recipients: {recipient_summary}"
                            )
                else:
                    if is_stable:
                        text = f"Low match / Unknown | Confidence: {confidence:.1f}"
                    else:
                        text = "Scanning..."
                    color = (0, 215, 255) if not is_stable else (20, 20, 255)

                new_recognized_faces.append(
                    {
                        "box": (x, y, w, h),
                        "text": text,
                        "color": color,
                        "confidence": confidence,
                    }
                )

            recognized_faces = new_recognized_faces
            recognition_history = next_recognition_history

        cv2.putText(
            frame,
            f"Tracking up to {MAX_MONITORED_FACES} faces",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 0),
            2,
        )

        for recognized_face in recognized_faces:
            x, y, w, h = recognized_face["box"]
            color = recognized_face["color"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                recognized_face["text"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {recognized_face['confidence']:.1f}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow(window_name, frame)
        if should_close_window(window_name):
            break

    camera.release()
    cv2.destroyAllWindows()


def prompt_student_details():
    student_id = input("Enter student ID: ").strip()
    name = input("Enter student name: ").strip()
    guardian_name = input("Enter parent/guardian name: ").strip()
    guardian_phone = input(
        "Enter parent/guardian phone number(s) (comma-separated if more than one): "
    ).strip()
    course = input("Enter student course/branch (example: CS, IT, AIML): ").strip()
    year = input("Enter student year (example: 1, 2, 3, 4): ").strip()
    class_name = input(
        "Enter student class/group/section label (example: CS2 or A): "
    ).strip()
    available_sections = ", ".join(get_sections().keys())

    if not student_id or not name:
        raise ValueError("Student ID and student name are required.")
    if not course:
        raise ValueError("Student course/branch is required.")
    if not year:
        raise ValueError("Student year is required.")

    candidate_section = class_name or f"{course}{year}"
    resolved_section = match_section_name(candidate_section)
    if get_sections() and not resolved_section:
        raise ValueError(
            f"No timetable section matched '{candidate_section}'. "
            f"Available sections: {available_sections}"
        )

    return (
        student_id,
        name,
        guardian_name,
        guardian_phone,
        course.upper(),
        class_name.upper(),
        year,
        resolved_section,
    )


def prompt_monitor_section() -> str:
    sections = get_sections()
    if not sections:
        return ""

    default_section = get_default_section()
    options = ", ".join(sections.keys())
    value = input(
        f"Enter section to monitor [{default_section}] ({options}): "
    ).strip()
    resolved = resolve_section(value or default_section)
    if not resolved:
        raise ValueError("Invalid section selected.")
    return resolved


def print_menu() -> None:
    print("\nStudent Recognition System")
    print("1. Register student and capture face samples")
    print("2. Train recognition model")
    print("3. Start timetable-aware live monitoring for all students")
    print("4. Show timetable and monitoring status")
    print("5. Exit")


def show_monitoring_status() -> None:
    section = prompt_monitor_section()
    monitor, status_text, slot = get_monitoring_decision(section_name=section)
    print(f"\nTimetable file: {TIMETABLE_FILE}")
    print(f"Selected section: {section or 'default'}")
    print(status_text)
    if slot:
        print(
            "Active slot: "
            f"{slot.get('start', '--:--')} - {slot.get('end', '--:--')} | "
            f"{slot.get('activity', 'Unknown')}"
        )
    else:
        print("No active slot right now.")
    print(f"Monitoring enabled: {'Yes' if monitor else 'No'}")
    print("Available sections:", ", ".join(get_sections().keys()))
    print("Ignored activities:", ", ".join(sorted(IGNORED_ACTIVITIES)))
    print(
        "To enable teacher/parent alerts, set ALERT_WEBHOOK_URL in your environment."
    )
    teacher_contacts = get_teacher_recipients(section, slot)
    if teacher_contacts:
        print(
            "Teacher recipients for this lecture:",
            ", ".join(
                f"{contact['name']} ({contact['phone']})"
                for contact in teacher_contacts
            ),
        )
    else:
        print(f"Teacher recipient config file: {TEACHERS_FILE}")
    print(
        "To include a public image link in the alert payload, set ALERT_IMAGE_BASE_URL "
        "to a URL where alert images are hosted."
    )


def main() -> None:
    ensure_directories()

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                (
                    student_id,
                    name,
                    guardian_name,
                    guardian_phone,
                    course,
                    class_name,
                    year,
                    section,
                ) = prompt_student_details()
                capture_student_faces(
                    student_id,
                    name,
                    guardian_phone,
                    guardian_name,
                    course,
                    class_name,
                    year,
                    section,
                )
            elif choice == "2":
                train_model()
            elif choice == "3":
                recognize_faces("")
            elif choice == "4":
                show_monitoring_status()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, 4 or 5.")
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
