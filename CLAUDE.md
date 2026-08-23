# Instrucciones para Claude en este repo

## Antes de citar resultados de detección de caídas / inmovilidad / actividades

Este repo tiene **dos fuentes de resultados muy distintas** y es fácil confundirlas:

1. **Logs sueltos en `vision/*.log`, `fusion/*.log`, `vlm/*.log`** (ej. `evaluate_falls_run.log`,
   `recalibrate_run*.log`): son corridas locales de desarrollo, **gitignoreados**
   (`vision/*.log` en `.gitignore`), **no versionados**, y pueden ser de mucho antes del
   último fix. No tienen fecha visible en git — hay que mirar el `mtime` del archivo
   (`ls -la`) y compararlo con `git log` de los scripts/READMEs relevantes antes de
   asumir que reflejan el estado actual.

2. **`vision/README.md`, `fusion/README.md`, `runner/README.md`, `vlm/README.md`**: acá
   sí se documentan los resultados calibrados y vigentes, con fecha de cada hallazgo.
   **Esta es la fuente de verdad.** Si un log local contradice el README, confiar en el
   README (o re-correr el script) antes de reportarle algo al usuario.

**Caso real que ya pasó:** se le reportó al usuario "340 falsos positivos en detección
de caídas" leyendo `vision/evaluate_falls_run.log` (11 de agosto, pre-fix, solo
heurística geométrica sin confirmación CLIP). El fix real ya estaba integrado desde el
commit `9356c15` (13 de agosto) y el número vigente, documentado en `vision/README.md`,
es **84.8% recall con 30 FP** sobre 500 eventos/292 clips (probe CLIP entrenado), y
**99.6% de los falsos positivos geométricos eliminados** (269/270) en validación
end-to-end sobre ~80 min de video doméstico continuo. El usuario tuvo que corregir a
Claude. Antes de citar un número de un `.log` suelto, verificar con
`git log --oneline -- <script/README relacionado>` que no haya un fix posterior.

## Estado actual verificado (2026-08-23), para no tener que re-derivarlo

- **Detección de actividades** (Cooking/EatingDrinkingMedication/Moving/Resting,
  fusión esqueleto GRU + CLIP, ver `fusion/README.md`): CLIP solo 75.8-87.6% según
  split, esqueleto solo 78.0-98.5%, híbrido mejora sobre ambos (82.8-99.2%). El split
  holdout (273 videos, el más realista) da el número más bajo — no citar el número de
  train/val como si fuera el esperable en producción.
- **Detección de caídas** (`vision/README.md`): con el probe CLIP de confirmación
  integrado, 84.8% recall / 30 FP sobre 500 eventos out-of-fold; 99.6% de FP
  geométricos eliminados en validación end-to-end. Pendiente documentado (no bug): en
  3 de 4 clips largos de `ToyotaSmarthome_FP` la inmovilidad no se dispara porque la
  persona nunca está 180s realmente quieta en el clip completo.
- **Inmovilidad/sedentarismo**: bug de historial corregido en `9356c15` (ventana de
  2.5s de la velocidad de caída se compartía indebidamente con el tracking de
  inmovilidad). Bloqueante identificado y no resuelto: `yolo11n-pose` no detecta bien
  a una persona acostada en sillón visto desde cámara ancha/elevada (94.5% de frames
  sin detección).

Antes de dar por buena cualquiera de estas cifras en una conversación nueva, confirmar
que no haya commits posteriores que las hayan cambiado (`git log` sobre los README de
cada carpeta).
