from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def find_soffice() -> str:
    candidates = [
        shutil.which("soffice.com"),
        shutil.which("soffice.exe"),
        r"E:\tools\LibreOffice\program\soffice.com",
        r"E:\tools\LibreOffice\program\soffice.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("LibreOffice soffice executable was not found.")


def find_pdftoppm() -> str:
    candidates = [
        shutil.which("pdftoppm"),
        r"C:\Users\j\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe",
        r"D:\texlive\2023\bin\windows\pdftoppm.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("pdftoppm executable was not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the final DOCX report to PDF and PNG pages.")
    parser.add_argument("--docx", default="reports/ml_power_report_academic.docx")
    parser.add_argument("--pdf-dir", default="reports/lo_pdf")
    parser.add_argument("--png-dir", default="reports/rendered_academic_fixed")
    parser.add_argument("--profile-dir", default="reports/lo_profile")
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docx = Path(args.docx).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    png_dir = Path(args.png_dir).resolve()
    profile_dir = Path(args.profile_dir).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    soffice = find_soffice()
    profile_uri = profile_dir.as_uri()
    expected_pdf = pdf_dir / f"{docx.stem}.pdf"
    if expected_pdf.exists():
        expected_pdf.unlink()
    subprocess.run(
        [
            soffice,
            "--headless",
            "--invisible",
            "--nodefault",
            "--nolockcheck",
            "--norestore",
            "--nofirststartwizard",
            f"-env:UserInstallation={profile_uri}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_dir),
            str(docx),
        ],
        check=True,
    )

    pdf = expected_pdf
    if not pdf.exists():
        raise FileNotFoundError(f"Expected PDF was not created: {pdf}")

    for old in png_dir.glob("*.png"):
        old.unlink()
    pdftoppm = find_pdftoppm()
    subprocess.run(
        [pdftoppm, "-png", "-r", str(args.dpi), str(pdf), str(png_dir / "page")],
        check=True,
    )
    pages = sorted(png_dir.glob("*.png"))
    print(f"PDF: {pdf}")
    print(f"Rendered pages: {len(pages)} -> {png_dir}")


if __name__ == "__main__":
    main()
