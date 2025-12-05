# vSphere GPU Check & Zuweisung (A100 / vGPU / Passthrough)

Ziel: Zweite GPU/vGPU der Ubuntu-VM zuweisen, um RAG-Dienste zu entlasten.

## 1) Über vSphere UI prüfen
1. VM ausschalten.
2. VM > Edit Settings > Gerät hinzufügen:
   - **Shared PCI Device** (für vGPU) → Profil wählen (z. B. A100-80C-40c oder 20c).
   - **PCI Device** (für Passthrough einer ganzen GPU), falls GPU als Passthrough bereitsteht.
3. Speichereinstellung: Memory Reservation = 100% (vGPU-Anforderung).
4. Speichern, VM starten.

### Wann Host-Reboot nötig ist?
- Wenn GPU aktuell als Passthrough markiert ist und auf vGPU umgestellt werden soll.
- Wenn GPU gerade einer anderen VM/Profil zugewiesen ist und freigegeben werden muss.
- Reine vGPU-Zuordnung bei freiem Profil braucht i.d.R. keinen Host-Reboot.

## 2) Host per SSH prüfen (ESXi)
```bash
# Liste der GPUs / Geräte
lspci | grep -i nvidia

# vGPU-Profile und Zuordnungen anzeigen
esxcfg-vgpu -l

# Passthrough-Status
esxcli hardware pci list | grep -i -A2 nvidia
esxcli hardware pci pcipassthru list | grep -i nvidia -A3
```

## 3) Entscheidungsmatrix
- **Freies vGPU-Profil verfügbar & Host nicht im Passthrough-Modus**  
  → VM aus, vGPU hinzufügen, VM starten (kein Host-Reboot).
- **GPU aktuell Passthrough, soll vGPU werden**  
  → Passthrough deaktivieren, Host-Reboot, dann vGPU-Profil zuweisen.
- **Zweite physische GPU vorhanden**  
  → Per Passthrough oder zweites vGPU-Profil der zweiten GPU hinzufügen (VM aus, ggf. Host-Reboot bei Umschaltung).

## 4) Nach Zuweisung in der VM
```bash
nvidia-smi           # sollten 2 GPUs/VGPU-Devices zeigen
```

## 5) Compose-Konfiguration anpassen (wenn 2 GPUs sichtbar)
Datei: `deploy/compose/.env.single-a100`
```
LLM_MS_GPU_ID=0
EMBEDDING_MS_GPU_ID=1
RANKING_MS_GPU_ID=1
YOLOX_MS_GPU_ID=0
YOLOX_GRAPHICS_MS_GPU_ID=0
YOLOX_TABLE_STRUCTURE_MS_GPU_ID=0
PADDLE_MS_GPU_ID=0
```
Dann NIMs neu starten:
```bash
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml up -d
```
