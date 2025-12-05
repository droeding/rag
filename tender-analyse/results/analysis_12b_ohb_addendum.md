# OHB RFI – Addendum (Bieterfragen, PoC, SLA, Referenzen)
**Stand:** 30.11.2025  
**Zweck:** Konkrete Anschlussartefakte für das Angebotspaket

## 1) Präzisierte Bieterfragen (VS-NfD/Operation)
- Welche Produktpräferenzen sind gesetzt für: SIEM/SOC, EDR/AV, Patch-Automatisierung? (bitte Hersteller/Version nennen)
- Ist der Betrieb in eingestuften Umgebungen (VS-NfD) vorgesehen? Falls ja: geforderte Zertifikate/Personen-SÜ/Hosting-Standorte?
- Gewünschtes ITSM-Target: Jira Service Management bleibt führend? Schnittstellen/Events/Schemas gewünscht?
- SOC-/SIEM-Integration: gewünschte Use-Case-Bibliothek, Alarmierungs- und Eskalationspfade (Zeitziele, Rollen)?
- Datenresidenz/Cloud: dürfen M365-/Sentinel-Daten in EU-Regionen gehalten werden? Vorgaben zu Sovereign Cloud?

## 2) PoC-Vorschlag (2–3 Wochen)
- Ziel: Nachweis Integrationsfähigkeit & Betriebsreife für EDR+SIEM+Patch-Automation.
- Umfang:
  - Anbindung von 5–10 Endpoints (Client/Server) an EDR (Defender/SentinelOne) inkl. Policies.
  - SIEM-Feed in Sentinel + 3 Beispiel-Use-Cases (Malware, Privilege Abuse, VPN-Anomalie) inkl. Dashboard/Alerts.
  - Patch-Automation-Flow (SCCM/WSUS/Patch my PC) mit Rollback und Reporting.
  - ITSM-Integration: Ticket-Auto-Create in Jira bei High-Severity-Events.
  - Kurzer Hardening-Check (BIT/Baseline) für einen repräsentativen Client/Server.
- Deliverables: PoC-Plan, Success-Kriterien, Abschlussreport mit Aufwandsschätzung Run/Build.

## 3) SLA-/Eskalationsskizze (Kurzform)
- Incident Triage: P1 ≤ 15 Min, P2 ≤ 30 Min, P3 ≤ 4h (24/7 für Security Events; ggf. 8x5 für Non-Security).
- Resolution Targets (Startpunkte, im RfP feinjustieren): P1 ≤ 4h; P2 ≤ 8h; P3 ≤ 2 BD; P4 nach Vereinbarung.
- Patch-Kadenz: kritische Security ≤ 48h (Pilot + gestaffelter Rollout), Standard monatlich.
- Reporting: Monatsreport (SLAs, Incidents, Changes, Patch-Compliance, Use-Case-Hit-Rates).
- Eskalation: 3 Stufen (Ops Lead → Service Manager → Engagement Exec); klarer SPOC auf Kundenseite.

## 4) Referenzen (Platzhalter zum Befüllen)
- SOC/SIEM (öffentlicher Sektor / VS-NfD-nah): \<Projekt, Jahr, Umfang, Tech (Sentinel/QRadar), SLA\>
- EDR/Endpoint (Defender/SentinelOne) Rollout & Betrieb: \<Projekt, Seats, Automationsgrad\>
- Patch-/Lifecycle-Automation (SCCM/Intune/WSUS/Patch my PC): \<Projekt, KPI (Compliance %, Mean Time to Patch)\>
- M365/Entra Security & Compliance (Purview/Defender for O365): \<Projekt, Scope, Besonderheiten Datenresidenz\>

## 5) Nächste operative Schritte
- Bieterfragen einreichen (Frist beachten, 22.08.2025 RFI-Deadline).
- PoC-Scope abstimmen und Termine vorschlagen.
- SLA-Entwurf + Eskalationsmatrix an OHB teilen.
- Referenzsteckbriefe befüllen (oben) und in das Angebotspaket aufnehmen.
