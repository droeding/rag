# Strategische Analyse – OHB RFI „Strategic IT Partner“ (test9)
**Datum:** 30.11.2025  
**Quelle:** RAG-Lauf mit LLM 12B (vLLM), Collection test9 (113 Entities), max_tokens 1024, Reranker aktiv, Kontext 128k  
**Pfad:** `tender-analyse/results/analysis_12b_test9_clean.md`

---

## 1. Management Summary
- OHB sucht einen strategischen IT-Partner; RFI-Phase, RfP folgt im **September 2025**.  
- Antworten auf das RFI bis **22.08.2025, 16:00 Uhr**.  
- Fokusbereiche: Security Monitoring (IBM QRadar), Endpoint Protection, Microsoft 365 Roadmap, Netzwerk & Remote Access, IT‑Service-Betrieb.  
- Datenlage ist gut strukturiert (Tabellen); RAG-Extraktion lieferte durchgängig präzise Ergebnisse (vgl. Benchmark 7/7).  

### Go / No-Go Indikatoren
**Go**
1. Klar umrissene Security- & Infrastruktur-Themen (QRadar, M365, Firewalls) passen zu Standard-Portfolio.  
2. Langfristiges Betriebs-/Service-Potenzial (Security Monitoring, Remote Access, M365-Roadmap bis 2026).  
3. Ansprechpartner klar benannt (Cyber Security Lead), erleichtert Dialog / Bieterfragen.

**No-Go / Risiken**
1. Endpoint-Produktnamen im RFI teils generisch → Gefahr, dass konkrete Produktpräferenzen erst im RfP spezifiziert werden.  
2. Remote-Access-/Firewall-Mix mehrerer Hersteller → Integrations- und Betriebsrisiko (Policies, Logging, Updates).  
3. Noch keine vertragliche Grundlage (EVB-Typ offen); RfP könnte Festpreis vs. agiles Vorgehen regeln – früh klären.

## 2. Steckbrief (Ausschreibungs-Fingerabdruck)
| Parameter                | Wert / Befund                          | Implikation |
|--------------------------|----------------------------------------|-------------|
| Auftraggeber             | OHB (RFI „Strategic IT Partner“)       | Public/Industrie, sicherheitskritisch |
| Verfahrensstand          | RFI, RfP geplant Sept 2025             | Früher Dialog möglich |
| Angebotsfrist RFI        | 22.08.2025, 16:00                      | Zeitkritisch für Rückfragen |
| Kontakt                  | Philipp Wehling, philipp.wehling@ohb.de| Klarer SPOC Security |
| Techn. Schwerpunkte      | Security Monitoring, Endpoint, M365, Netzwerk/RA | Breit, aber klar priorisiert |
| Geplanter Betrieb        | Mehrjährige Services erwartet          | Chance für Recurring Revenue |

## 3. Kernergebnisse aus dem RAG-Run (test9)
### 3.1 Security Monitoring / SIEM
- **IBM QRadar On-Prem** als SIEM.  
- Operative Tools im Stack: **JIRA Service Management**, **iDoIt CMDB**, **VMRay Sandboxing**, **ASGARD**, **DCSO**.

### 3.2 Endpoint Protection
- Funktionen: Malware/Ransomware-Schutz, Attack Surface Reduction, Gerätekontrolle, Integration mit M365 Defender.  
- Produktnennung im RFI nicht eindeutig; wahrscheinlich M365 Defender + ggf. SentinelOne (zu klären im RfP/Bieterfrage).

### 3.3 Netzwerk & Remote Access
- Firewalls: **PaloAlto**, **FortiGate**, **Genua**, **SINA**.  
- Remote Access: **GlobalProtect**, **Aruba**, **ECOS**, **Genua RA**, **SINA WAN**.  
- Implikation: Heterogene Policy-/Logging-Landschaft; Vereinheitlichung oder Betriebs-Playbook anbieten.

### 3.4 Microsoft 365 Roadmap
- **Aktuell:** SharePoint Online, Teams, Entra ID, Defender for O365, Intune.  
- **2026 geplant:** Purview Compliance, Power Platform, Exchange Online, Teams-Telefonie.  
- Empfehlung: Lizenz-/Governance-Angebot plus Begleitung Purview/Teams-Telefonie.

### 3.5 Termine / Zeitplan
- **RFI-Antwort:** 22.08.2025, 16:00 Uhr.  
- **RfP-Veröffentlichung:** September 2025.

### 3.6 Kontakt
- **Philipp Wehling** – Cyber Security (E-Mail s.o.).

### 3.7 Servicebereiche (aus Tabellen aggregiert)
- NOC (Firewalls, WAN/RA), Endpoint Protection, Security Monitoring, Server/App-Management, WLAN/Firewall-Operations.

## 4. Risiken & Mitigation
| Risiko | Beschreibung | Mitigation |
|---|---|---|
| Produktpräferenz unklar (Endpoint) | RFI nennt Features statt Produkte | Bieterfrage: bevorzugte EDR/AV-Produkte? Vorschlag: M365 Defender + SentinelOne als Alternative |
| Multi-Vendor FW/RA | Unterschiedliche Policies/Logs, höherer Betriebsaufwand | Betriebs-Playbook + SIEM-Use-Cases pro Hersteller anbieten |
| Vertrags-/EVB-Typ offen | Pricing- und Scope-Risiko im RfP | Frühzeitig EVB-/Vertragsrahmen erfragen; Festpreis nur mit klarer Abgrenzung, sonst Aufwand/Service-Modell |
| M365-Roadmap 2026 | Scope-Drift zwischen RFI/RfP möglich | Phasenplan: „Jetzt“ vs. „2026“ mit Change-Budget vorschlagen |

## 5. Empfehlungen (kurz)
1. **Bieterfragen** früh platzieren: EDR-Produktwahl, gewünschtes Betriebsmodell (SIEM Run/Build), EVB-Typ.  
2. **Angebotspositionierung:** „Sicherer Betrieb + Vereinheitlichung“ – Fokus auf Runbook, SIEM-Use-Cases, Hardening-Standards.  
3. **M365-Track** als eigener Workstream (Purview/Teams-Telefonie) mit klaren Meilensteinen bis 2026.  
4. **Referenzen** für SIEM/Firewall-Multivendor und Public-Sector-Security beilegen.  

## 6. Nächste Schritte
- Check interner Kapazitäten für SIEM & Network Ops (ab Q3 2025).  
- Kurze Exec-Summary (1 Pager) für Vertrieb erstellen.  
- Bieterfragen vorbereiten und vor Frist einreichen.  
- Optional: Angebot für PoC „Policy-/Logging-Harmonisierung“ vor RfP anbieten.

---

*Dieses Dokument wurde aus dem RAG-Ergebnis bereinigt und komprimiert. Keine Inhalte außerhalb der RFI-Datenbasis wurden hinzugefügt.*
