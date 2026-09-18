path = "models/base.py"

with open(path, "rb") as f:
    raw = f.read()

try:
    text = raw.decode("utf-16")
    print("Terdeteksi UTF-16, mendekode ulang...")
except UnicodeDecodeError:
    print("Bukan UTF-16 valid, membuang null byte secara paksa...")
    text = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Selesai. File ditulis ulang sebagai UTF-8 bersih.")