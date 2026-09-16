from pathlib import Path

DATA_DIR = Path("data")
OUTPUT = Path("data.txt")

files = sorted(DATA_DIR.glob("*.txt"))

with OUTPUT.open("w", encoding="utf-8") as out:
    for file in files:
        print(f"Adding: {file.name}")

        text = file.read_text(encoding="utf-8", errors="ignore")

        # Separate books clearly.
        out.write("\n\n")
        out.write("=" * 80)
        out.write("\n\n")
        out.write(text)
        out.write("\n\n")

text = OUTPUT.read_text(encoding="utf-8")

print()
print(f"Books: {len(files)}")
print(f"Characters: {len(text):,}")
print(f"Saved: {OUTPUT}")