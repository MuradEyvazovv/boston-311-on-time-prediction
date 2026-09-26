"""Download the Boston 311 service-request CSVs from Analyze Boston (data.boston.gov).

The resources are discovered through the official CKAN API (package_show on the
"311-service-requests" dataset) rather than hard-coded URLs, because the City
re-uploads files and the download URLs change. We fetch:

  * "311 Service Requests - 2024"        (legacy system; used ONLY as look-back
                                          history for the 30-day backlog features
                                          of early-January 2025 cases)
  * "311 Service Requests - 2025"        (legacy Lagan system, full year 2025)
  * "311 Service Requests - 2026"        (legacy Lagan system, 2026 to date)
  * "311 Service Requests - NEW SYSTEM"  (new BCS system, which started taking
                                          over case types from late 2025)
  * the NEW SYSTEM data dictionary PDF   (for reference only)

Everything is saved under data/raw/ (gitignored) together with a manifest.json
that records the CKAN resource id, the portal's last_modified stamp, the byte
size and a SHA-256 hash, so a later run can tell whether the upstream data changed.

Usage:
    python src/download.py            # skip files that already exist
    python src/download.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CKAN = "https://data.boston.gov/api/3/action"
PACKAGE_ID = "311-service-requests"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# CKAN resource name -> local file name
WANTED = {
    "311 Service Requests - 2024": "legacy_2024.csv",
    "311 Service Requests - 2025": "legacy_2025.csv",
    "311 Service Requests - 2026": "legacy_2026.csv",
    "311 Service Requests - NEW SYSTEM": "new_system.csv",
}
DICTIONARY_SUFFIX = "311-service-requests-data-dictionary-new-system.pdf"
DICTIONARY_FILE = "data_dictionary_new_system.pdf"

UA = {"User-Agent": "boston-311-on-time-prediction/1.0 (portfolio project)"}


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context.

    python.org builds of Python on macOS ship without a CA bundle unless
    "Install Certificates.command" was run, so fall back to certifi (if
    installed) or the macOS system bundle. Verification is never disabled.
    """
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for cafile in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/cert.pem"):
        if cafile and Path(cafile).exists():
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


SSL_CTX = _ssl_context()


class _StripDefaultPortRedirect(urllib.request.HTTPRedirectHandler):
    """The portal 302-redirects downloads to a pre-signed S3 URL of the form
    https://s3.amazonaws.com:443/...; the signature covers the Host header, and
    urllib would send "Host: s3.amazonaws.com:443" (-> 403). Drop the default port."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newurl = newurl.replace("https://s3.amazonaws.com:443/", "https://s3.amazonaws.com/")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=SSL_CTX), _StripDefaultPortRedirect()
)


def ckan(action: str, **params) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{CKAN}/{action}?{query}", headers=UA)
    with OPENER.open(req, timeout=60) as resp:
        payload = json.load(resp)
    if not payload.get("success"):
        raise RuntimeError(f"CKAN call {action} failed: {payload}")
    return payload["result"]


def download(url: str, dest: Path) -> tuple[int, str]:
    """Stream url to dest, return (bytes, sha256)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    sha = hashlib.sha256()
    n = 0
    req = urllib.request.Request(url, headers=UA)
    with OPENER.open(req, timeout=300) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            sha.update(chunk)
            n += len(chunk)
            print(f"\r  {dest.name}: {n / 1e6:8.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return n, sha.hexdigest()


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pkg = ckan("package_show", id=PACKAGE_ID)
    print(f"Dataset: {pkg['title']}  |  license: {pkg.get('license_title')}")

    targets = []
    for res in pkg["resources"]:
        name = (res.get("name") or "").strip()
        if name in WANTED:
            targets.append((res, WANTED[name]))
        elif res.get("url", "").endswith(DICTIONARY_SUFFIX):
            targets.append((res, DICTIONARY_FILE))
    missing = set(WANTED) - {(r.get("name") or "").strip() for r, _ in targets}
    if missing:
        print(f"ERROR: resources not found on the portal: {sorted(missing)}", file=sys.stderr)
        return 1

    manifest = {
        "source": f"https://data.boston.gov/dataset/{PACKAGE_ID}",
        "license": pkg.get("license_title"),
        "license_url": pkg.get("license_url"),
        "files": [],
    }
    t0 = time.time()
    for res, fname in targets:
        dest = RAW_DIR / fname
        if dest.exists() and not args.force:
            print(f"  {fname}: exists, skipping (use --force to refresh)")
            size, digest = dest.stat().st_size, sha256_of(dest)
            downloaded_at = datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc).isoformat()
        else:
            size, digest = download(res["url"], dest)
            downloaded_at = datetime.now(timezone.utc).isoformat()
        manifest["files"].append({
            "file": f"data/raw/{fname}",
            "resource_name": res.get("name"),
            "resource_id": res["id"],
            "url": res["url"],
            "portal_last_modified": res.get("last_modified"),
            "downloaded_at_utc": downloaded_at,
            "bytes": size,
            "sha256": digest,
        })

    with open(RAW_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(f["bytes"] for f in manifest["files"])
    print(f"Done: {len(manifest['files'])} files, {total / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    print(f"Manifest: {RAW_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
