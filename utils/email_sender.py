"""Email sender for prediction and training results.

Credentials are read from environment variables (or a .env file):
    ASP_EMAIL_PASSWORD   – SMTP password / Gmail app password
    ASP_EMAIL_USERNAME   – (optional) overrides config username
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load a .env file from cwd (primary) or source tree (fallback)."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def send_prediction_email(config: dict, prediction_data: dict) -> bool:
    """Send prediction results via email.

    Parameters
    ----------
    config : dict
        The full app config (must contain an ``email`` section).
    prediction_data : dict
        Keys: prediction_date, base_date, base_price, probability,
              decision, confidence_level, market_regime, model_version,
              feature_details (list[dict]).

    Returns
    -------
    bool
        True if the email was sent successfully.
    """
    email_cfg = config.get("email", {})
    if not email_cfg.get("enabled", False):
        return False

    _load_dotenv()

    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com") or "smtp.gmail.com"
    smtp_port = email_cfg.get("smtp_port", 587)
    username = os.environ.get("ASP_EMAIL_USERNAME", email_cfg.get("username", email_cfg.get("from", "")))
    password = os.environ.get("ASP_EMAIL_PASSWORD", email_cfg.get("password", ""))
    email_from = email_cfg.get("from", username)
    email_to = email_cfg.get("to", "")

    if not all([smtp_server, username, password, email_to]):
        logger.warning("Email not configured — skipping send.")
        return False

    # ── Build email content ──────────────────────────────────────
    prob = prediction_data["probability"]
    dec = prediction_data["decision"]
    pred_date = prediction_data["prediction_date"]
    base_date = prediction_data["base_date"]
    base_price = prediction_data["base_price"]
    base_return = prediction_data.get("base_return", 0.0)
    confidence = prediction_data["confidence_level"]
    regime = prediction_data["market_regime"]
    model_ver = prediction_data["model_version"]
    features = prediction_data.get("feature_details", [])[:15]

    direction = "UP" if prob > 0.5 else "DOWN"
    colour = "#2e7d32" if prob > 0.5 else "#c62828"

    subject = (
        f"ASX200 Prediction {pred_date}: {direction} "
        f"({prob*100:.1f}%) — {confidence}"
    )

    # Plain-text fallback
    plain_lines = [
        f"ASX200 Prediction for {pred_date}",
        f"",
        f"Base date:   {base_date}",
        f"Base price:  {base_price:,.2f} ({base_return:+.2f}%)",
        f"P(up):       {prob*100:.1f}%",
        f"P(down):     {(1-prob)*100:.1f}%",
        f"Decision:    {dec}",
        f"Confidence:  {confidence}",
        f"Regime:      {regime}",
        f"Model:       {model_ver}",
        f"",
        f"Top features:",
    ]
    for fd in features:
        val = fd["value"]
        imp = fd["importance"]
        name = fd["name"]
        is_pct = ('return' in name or 'change' in name
                  or 'premium' in name or name == 'macd_histogram')
        if is_pct:
            plain_lines.append(f"  {name:<28} {val*100:>+10.2f}%  imp {imp:.4f}")
        else:
            plain_lines.append(f"  {name:<28} {val:>+12.4f}  imp {imp:.4f}")
    plain_body = "\n".join(plain_lines)

    # HTML body
    feature_rows = ""
    for fd in features:
        val = fd["value"]
        imp = fd["importance"]
        name = fd["name"]
        is_pct = ('return' in name or 'change' in name
                  or 'premium' in name or name == 'macd_histogram')
        if is_pct:
            val_str = f"{val*100:+.2f}%"
        else:
            val_str = f"{val:.4f}"
        feature_rows += (
            f"<tr><td style='padding:4px 8px'>{name}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{val_str}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{imp:.4f}</td></tr>\n"
        )

    html_body = f"""\
<html>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
<h2 style="color:{colour}">ASX200 Prediction for {pred_date}</h2>
<table style="border-collapse:collapse;width:100%">
  <tr><td style="padding:4px 8px"><b>Base date</b></td>
      <td style="padding:4px 8px">{base_date}</td></tr>
  <tr><td style="padding:4px 8px"><b>ASX200 price</b></td>
      <td style="padding:4px 8px">{base_price:,.2f} ({base_return:+.2f}%)</td></tr>
  <tr><td style="padding:4px 8px"><b>P(positive)</b></td>
      <td style="padding:4px 8px;color:{colour};font-weight:bold">{prob*100:.1f}%</td></tr>
  <tr><td style="padding:4px 8px"><b>P(negative)</b></td>
      <td style="padding:4px 8px">{(1-prob)*100:.1f}%</td></tr>
  <tr><td style="padding:4px 8px"><b>Decision</b></td>
      <td style="padding:4px 8px;color:{colour};font-weight:bold">{dec}</td></tr>
  <tr><td style="padding:4px 8px"><b>Confidence</b></td>
      <td style="padding:4px 8px">{confidence}</td></tr>
  <tr><td style="padding:4px 8px"><b>Market regime</b></td>
      <td style="padding:4px 8px">{regime}</td></tr>
  <tr><td style="padding:4px 8px"><b>Model version</b></td>
      <td style="padding:4px 8px">{model_ver}</td></tr>
</table>

<h3>Top 15 Features</h3>
<table style="border-collapse:collapse;width:100%;font-size:0.9em">
  <tr style="background:#f5f5f5">
    <th style="padding:4px 8px;text-align:left">Feature</th>
    <th style="padding:4px 8px;text-align:right">Value</th>
    <th style="padding:4px 8px;text-align:right">Importance</th>
  </tr>
  {feature_rows}
</table>

<p style="color:#999;font-size:0.8em;margin-top:20px">
  Sent by AusSuperPredictor v{model_ver}
</p>
</body>
</html>
"""

    # ── Send ─────────────────────────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(username, password)
            server.sendmail(email_from, [email_to], msg.as_string())
        logger.info("Prediction email sent to %s", email_to)
        return True
    except Exception as exc:
        logger.warning("Failed to send prediction email: %s", exc)
        return False


def _get_smtp_credentials(config: dict):
    """Extract SMTP credentials from config + env. Returns None if incomplete."""
    email_cfg = config.get("email", {})
    if not email_cfg.get("enabled", False):
        return None

    _load_dotenv()

    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com") or "smtp.gmail.com"
    smtp_port = email_cfg.get("smtp_port", 587)
    username = os.environ.get("ASP_EMAIL_USERNAME",
                              email_cfg.get("username", email_cfg.get("from", "")))
    password = os.environ.get("ASP_EMAIL_PASSWORD", email_cfg.get("password", ""))
    email_from = email_cfg.get("from", username)
    email_to = email_cfg.get("to", "")

    if not all([smtp_server, username, password, email_to]):
        logger.warning("Email not configured — skipping send.")
        return None

    return {
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
        "username": username,
        "password": password,
        "email_from": email_from,
        "email_to": email_to,
    }


def _send(creds: dict, subject: str, plain_body: str, html_body: str) -> bool:
    """Low-level SMTP send helper."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = creds["email_from"]
    msg["To"] = creds["email_to"]
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(creds["smtp_server"], creds["smtp_port"], timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(creds["username"], creds["password"])
            server.sendmail(creds["email_from"], [creds["email_to"]], msg.as_string())
        return True
    except Exception as exc:
        logger.warning("Failed to send email: %s", exc)
        return False


def send_training_email(config: dict, training_data: dict) -> bool:
    """Send training results via email.

    Parameters
    ----------
    config : dict
        The full app config (must contain an ``email`` section).
    training_data : dict
        Keys: train_accuracy, test_accuracy, feature_importance (list[dict]),
              calibration (dict with expected_calibration_error, max_calibration_error),
              prev (dict | None) – the full previous training snapshot.

    Returns
    -------
    bool
        True if the email was sent successfully.
    """
    creds = _get_smtp_credentials(config)
    if creds is None:
        return False

    train_acc = training_data["train_accuracy"]
    test_acc = training_data["test_accuracy"]
    gap = train_acc - test_acc
    features = training_data.get("feature_importance", [])[:10]
    cal = training_data.get("calibration", {})
    ece = cal.get("expected_calibration_error")
    mce = cal.get("max_calibration_error")
    prev = training_data.get("prev") or {}
    prev_test = prev.get("test_accuracy")
    prev_train = prev.get("train_accuracy")
    prev_cal = prev.get("calibration", {})
    prev_feats = {f["feature"]: f["importance"] for f in prev.get("feature_importance", [])}

    # Colour based on test accuracy
    colour = "#2e7d32" if test_acc >= 0.65 else "#e65100" if test_acc >= 0.55 else "#c62828"

    delta_str = ""
    if prev_test is not None:
        delta = test_acc - prev_test
        delta_str = f" ({delta:+.1%})"

    subject = f"ASX200 Model Training: Test {test_acc:.1%}{delta_str}"

    # ── Plain text ──────────────────────────────────────────────
    plain_lines = [
        "ASX200 Model Training Results",
        "",
        f"Train accuracy:  {train_acc:.1%}",
        f"Test accuracy:   {test_acc:.1%}",
        f"Overfit gap:     {gap:.1%}",
    ]
    if ece is not None:
        plain_lines.append(f"ECE:             {ece:.4f}")
    if mce is not None:
        plain_lines.append(f"MCE:             {mce:.4f}")
    plain_lines += ["", "Top features:"]
    for fd in features:
        plain_lines.append(f"  {fd['feature']:<28} {fd['importance']:.4f}")

    # Delta section (plain) — unified arrow format
    if prev_test is not None:
        def _arrow(d, threshold=0.0005):
            return "▲" if d > threshold else ("▼" if d < -threshold else "─")

        d_train = train_acc - (prev_train or 0)
        d_test = test_acc - prev_test
        prev_ece = prev_cal.get("ece")
        prev_mce = prev_cal.get("mce")

        plain_lines += ["", "── Changes from previous model ──"]
        plain_lines.append(f"    {_arrow(d_train)} Train acc:  {train_acc:.4f} ({d_train:+.4f})")
        plain_lines.append(f"    {_arrow(d_test)} Test acc:   {test_acc:.4f} ({d_test:+.4f})")
        if ece is not None and prev_ece is not None:
            d_ece = ece - prev_ece
            d_mce = (mce or 0) - (prev_mce or 0)
            # For ECE/MCE lower is better, so invert arrow
            plain_lines.append(f"    {_arrow(-d_ece)} ECE:        {ece:.4f} ({d_ece:+.4f})")
            plain_lines.append(f"    {_arrow(-d_mce)} MCE:        {(mce or 0):.4f} ({d_mce:+.4f})")
        if prev_feats:
            for fd in features:
                name = fd["feature"]
                new_imp = fd["importance"]
                old_imp = prev_feats.get(name, 0)
                d = new_imp - old_imp
                plain_lines.append(f"    {_arrow(d, 0.005)} {name}: {new_imp:.4f} ({d:+.4f})")

    plain_body = "\n".join(plain_lines)

    # ── HTML ────────────────────────────────────────────────────
    feature_rows = ""
    for i, fd in enumerate(features):
        bg = " style='background:#f5f5f5'" if i % 2 == 0 else ""
        feature_rows += (
            f"<tr{bg}><td style='padding:4px 8px'>{fd['feature']}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{fd['importance']:.4f}</td></tr>\n"
        )

    cal_rows = ""
    if ece is not None:
        cal_rows += (f"<tr><td style='padding:4px 8px'><b>ECE</b></td>"
                     f"<td style='padding:4px 8px'>{ece:.4f}</td></tr>")
    if mce is not None:
        cal_rows += (f"<tr><td style='padding:4px 8px'><b>MCE</b></td>"
                     f"<td style='padding:4px 8px'>{mce:.4f}</td></tr>")

    # Delta HTML section — single table with arrows, matching GUI format
    delta_html = ""
    if prev_test is not None:
        def _html_arrow(d, threshold=0.0005):
            if d > threshold:
                return "&#9650;"   # ▲
            elif d < -threshold:
                return "&#9660;"   # ▼
            return "&#9472;"       # ─

        def _delta_colour(d):
            if d > 0.0005:
                return "#2e7d32"
            elif d < -0.0005:
                return "#c62828"
            return "#555"

        d_train = train_acc - (prev_train or 0)
        d_test = test_acc - prev_test
        prev_ece = prev_cal.get("ece")
        prev_mce = prev_cal.get("mce")

        rows = []  # list of (arrow_html, label, value_str, delta_str, delta_colour)

        rows.append((_html_arrow(d_train), "Train acc", f"{train_acc:.4f}",
                     f"({d_train:+.4f})", _delta_colour(d_train)))
        rows.append((_html_arrow(d_test), "Test acc", f"{test_acc:.4f}",
                     f"({d_test:+.4f})", _delta_colour(d_test)))

        if ece is not None and prev_ece is not None:
            d_ece = ece - prev_ece
            d_mce = (mce or 0) - (prev_mce or 0)
            # Lower is better for ECE/MCE → invert arrow
            rows.append((_html_arrow(-d_ece), "ECE", f"{ece:.4f}",
                         f"({d_ece:+.4f})", _delta_colour(-d_ece)))
            rows.append((_html_arrow(-d_mce), "MCE", f"{(mce or 0):.4f}",
                         f"({d_mce:+.4f})", _delta_colour(-d_mce)))

        if prev_feats:
            for fd in features:
                name = fd["feature"]
                new_imp = fd["importance"]
                old_imp = prev_feats.get(name, 0)
                d = new_imp - old_imp
                rows.append((_html_arrow(d, 0.005), name, f"{new_imp:.4f}",
                             f"({d:+.4f})", _delta_colour(d)))

        delta_rows_html = ""
        for i, (arrow, label, val, dstr, dcol) in enumerate(rows):
            bg = " style='background:#f5f5f5'" if i % 2 == 0 else ""
            delta_rows_html += (
                f"<tr{bg}><td style='padding:4px 8px'>{arrow} {label}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{val}</td>"
                f"<td style='padding:4px 8px;text-align:right;color:{dcol}'>"
                f"{dstr}</td></tr>\n"
            )

        delta_html = f"""
<h3 style="margin-top:16px">Changes from previous model</h3>
<table style="border-collapse:collapse;width:100%;font-size:0.9em">
  <tr style="background:#e0e0e0">
    <th style="padding:4px 8px;text-align:left">Metric</th>
    <th style="padding:4px 8px;text-align:right">Value</th>
    <th style="padding:4px 8px;text-align:right">Delta</th>
  </tr>
  {delta_rows_html}
</table>"""

    html_body = f"""\
<html>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
<h2 style="color:{colour}">ASX200 Model Training Results</h2>
<table style="border-collapse:collapse;width:100%">
  <tr><td style="padding:4px 8px"><b>Train accuracy</b></td>
      <td style="padding:4px 8px">{train_acc:.1%}</td></tr>
  <tr><td style="padding:4px 8px"><b>Test accuracy</b></td>
      <td style="padding:4px 8px;color:{colour};font-weight:bold">{test_acc:.1%}</td></tr>
  <tr><td style="padding:4px 8px"><b>Overfit gap</b></td>
      <td style="padding:4px 8px">{gap:.1%}</td></tr>
  {cal_rows}
</table>

<h3>Top 10 Features</h3>
<table style="border-collapse:collapse;width:100%;font-size:0.9em">
  <tr style="background:#e0e0e0">
    <th style="padding:4px 8px;text-align:left">Feature</th>
    <th style="padding:4px 8px;text-align:right">Importance</th>
  </tr>
  {feature_rows}
</table>
{delta_html}
<p style="color:#999;font-size:0.8em;margin-top:20px">
  Sent by AusSuperPredictor
</p>
</body>
</html>
"""

    sent = _send(creds, subject, plain_body, html_body)
    if sent:
        logger.info("Training email sent to %s", creds["email_to"])
    return sent
