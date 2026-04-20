import os
from pathlib import Path
from typing import Iterable

from flask import Flask, jsonify, request
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def load_dotenv_file(file_path: Path) -> None:
    if not file_path.exists():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(BASE_DIR / ".env")


def env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def split_phone_numbers(value) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value or "").split(",")

    phone_numbers = []
    for raw_value in raw_values:
        phone = str(raw_value or "").strip()
        if phone:
            phone_numbers.append(phone)
    return phone_numbers


def normalize_phone_number(phone: str) -> str:
    digits = "".join(character for character in str(phone or "") if character.isdigit())
    if not digits:
        return ""
    if str(phone).strip().startswith("+"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    return f"+{digits}"


def dedupe_contacts(contacts: Iterable[dict]) -> list[dict]:
    unique_contacts = []
    seen = set()
    for contact in contacts:
        phone = normalize_phone_number(contact.get("phone", ""))
        if not phone or phone in seen:
            continue
        seen.add(phone)
        clean_contact = dict(contact)
        clean_contact["phone"] = phone
        unique_contacts.append(clean_contact)
    return unique_contacts


def get_twilio_client() -> Client | None:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if (
        not account_sid
        or not auth_token
        or "your_twilio" in account_sid.lower()
        or "your_twilio" in auth_token.lower()
    ):
        return None
    return Client(account_sid, auth_token)


def get_twilio_sender() -> str:
    sender = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
    if not sender or sender == "+9897057701":
        return ""
    return sender


def build_parent_contacts(payload: dict) -> list[dict]:
    contacts = []

    source_contacts = payload.get("parent_recipients")
    if not isinstance(source_contacts, list):
        source_contacts = payload.get("parent_contacts")

    if isinstance(source_contacts, list):
        for contact in source_contacts:
            if not isinstance(contact, dict):
                continue
            contacts.append(
                {
                    "name": contact.get("name", "Parent/Guardian"),
                    "phone": contact.get("phone", ""),
                    "role": "parent",
                }
            )

    guardian_phone = payload.get("guardian_phone", "")
    for phone in split_phone_numbers(guardian_phone):
        contacts.append(
            {
                "name": "Parent/Guardian",
                "phone": phone,
                "role": "parent",
            }
        )

    return dedupe_contacts(contacts)


def build_teacher_contacts(payload: dict) -> list[dict]:
    contacts = []
    source_contacts = payload.get("teacher_recipients")
    if not isinstance(source_contacts, list):
        source_contacts = payload.get("teacher_contacts")

    if isinstance(source_contacts, list):
        for contact in source_contacts:
            if not isinstance(contact, dict):
                continue
            contacts.append(
                {
                    "name": contact.get("name", "Faculty"),
                    "phone": contact.get("phone", ""),
                    "role": "teacher",
                }
            )
    return dedupe_contacts(contacts)


def send_sms_messages(contacts: list[dict], message: str) -> tuple[list[dict], list[dict]]:
    client = get_twilio_client()
    sender_number = get_twilio_sender()
    sent_messages = []
    failed_messages = []

    if not contacts:
        return sent_messages, failed_messages

    if client is None or not sender_number:
        for contact in contacts:
            failed_messages.append(
                {
                    "name": contact.get("name", ""),
                    "phone": contact.get("phone", ""),
                    "error": "Twilio credentials are not configured.",
                }
            )
        return sent_messages, failed_messages

    for contact in contacts:
        try:
            response = client.messages.create(
                body=message,
                from_=sender_number,
                to=contact["phone"],
            )
            sent_messages.append(
                {
                    "name": contact.get("name", ""),
                    "phone": contact["phone"],
                    "sid": response.sid,
                }
            )
        except TwilioRestException as error:
            failed_messages.append(
                {
                    "name": contact.get("name", ""),
                    "phone": contact["phone"],
                    "error": str(error),
                }
            )

    return sent_messages, failed_messages


@app.get("/")
def healthcheck():
    return jsonify(
        {
            "status": "ok",
            "service": "student-alert-webhook",
            "twilio_configured": bool(get_twilio_client() and get_twilio_sender()),
            "env_file_loaded": (BASE_DIR / ".env").exists(),
        }
    )


@app.post("/alert")
def alert():
    payload = request.get_json(silent=True) or {}

    parent_contacts = build_parent_contacts(payload)
    teacher_contacts = build_teacher_contacts(payload)

    parent_message = payload.get("parent_message") or payload.get("message") or "Student alert"
    teacher_message = payload.get("teacher_message") or payload.get("message") or "Student alert"

    sent = []
    failed = []

    if env_flag("SEND_PARENT_ALERTS", "1"):
        parent_sent, parent_failed = send_sms_messages(parent_contacts, parent_message)
        sent.extend([{**entry, "role": "parent"} for entry in parent_sent])
        failed.extend([{**entry, "role": "parent"} for entry in parent_failed])

    if env_flag("SEND_TEACHER_ALERTS", "1"):
        teacher_sent, teacher_failed = send_sms_messages(teacher_contacts, teacher_message)
        sent.extend([{**entry, "role": "teacher"} for entry in teacher_sent])
        failed.extend([{**entry, "role": "teacher"} for entry in teacher_failed])

    return jsonify(
        {
            "status": "ok" if not failed else "partial",
            "student_id": payload.get("student_id", ""),
            "student_name": payload.get("student_name", ""),
            "parent_contact_count": len(parent_contacts),
            "teacher_contact_count": len(teacher_contacts),
            "sent_count": len(sent),
            "failed_count": len(failed),
            "sent": sent,
            "failed": failed,
        }
    )


if __name__ == "__main__":
    host = os.getenv("ALERT_WEBHOOK_HOST", "127.0.0.1")
    port = int(os.getenv("ALERT_WEBHOOK_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
