# Benchmark: Tabellenextraktion aus OHB-RFI (30.11.2025)

Quelle: manueller 7-Fragen-Test im NVIDIA RAG Blueprint UI, Kollektion **OHB** (RFI-OHB_Strategic_IT_Partner.pdf). Vollständiger UI-Report: http://10.131.7.162:8090/ (intern).

## Ergebnisse (7/7 erfolgreich)
- **Endpoint Protection Technologien** – Features korrekt, Produktspezifika (Microsoft Defender, Sentinel One) teils unterschlagen.
- **SIEM & Security Monitoring Stack** – IBM QRadar OnPrem + JIRA SM, iDoIt CMDB, VMRay, ASGARD, DCSO. **Exzellent.**
- **Kontaktdaten** – Philipp Wehling, philipp.wehling@ohb.de. **Perfekt.**
- **M365 Dienste & Roadmap** – Aktuell: SPO, Teams, Entra ID, Defender for O365, Intune. 2026: Purview, Power Platform, Exchange Online, Teams-Telefonie. **Sehr gut.**
- **Zeitplan-Tabelle** – RFI-Antwort: 22.08.2025, 16:00; RfP: September 2025. **Perfekt.**
- **Netzwerk-Technologien** – Firewalls: PaloAlto, FortiGate, Genua, SINA; Remote Access: Global Protect, Aruba, ECOS, Genua RA, SINA WAN. **Perfekt.**
- **Cross-Domain-Aggregation** – Servicebereiche (NOC, Endpoint, Security Monitoring, Server/App, WLAN/Firewall) mit 15+ Technologien. **Herausragend.**

## Stärken
- Exzellente Tabellen-/Listenextraktion, Mehrfach-Entity-Retrieval und zeitliche Differenzierung („aktuell“ vs. „geplant“).
- Präzise Deadlines und Kontaktfelder; Technologie-Mappings korrekt (QRadar→SIEM, PaloAlto→Firewall).

## Optimierungspotenzial
- Bei generischen Feature-Fragen gelegentlich Feature-Beschreibung statt konkreter Produktnamen.
- Für sehr komplexe Vergleiche könnten Metadaten-Tags oder strukturierte Felder helfen.

## Gesamtbewertung
**9/10** – RAG beantwortet tabellenbasierte Fragen des OHB-RFI sehr zuverlässig, auch bei serviceübergreifender Aggregation.
