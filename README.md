# Multimodale Klassifikation keltischer Büscheldinaren

- Ziel: Kombination von Bild- und Textdaten zur automatischen Typklassifikation von 9 Typen (P, A-H)  
- Input: Bilddatensatz (~2000 Münzen) + Excel-Tabelle mit Beschreibungen für Vorder-/Rückseite  
- Ansatz: Modell für Bilder + BERT für Texte → kombinierte Embeddings → Klassifikation  
- Tools: PyTorch, pandas, scikit-learn, TBD

---

## Abhängigkeiten / Dependencies

Für das Projekt werden folgende Python-Pakete benötigt:

- pandas  
- scikit-learn  
- matplotlib  
- numpy  
- opencv-python-headless  
- pillow  
- hdbscan  
- pytorch-grad-cam  

---

## Installation mit CUDA-Unterstützung

Für GPU-Beschleunigung empfehlen wir die Installation von PyTorch und torchvision mit CUDA 11.8 Support:

```bash
pip install torch==2.7.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118


# Triplet-Datenset-Erstellung für Stempelbilder

Dieses Projekt erzeugt Trainingsdaten für Triplet-Learning aus Bilddaten keltischer Münzen (Vorder- und Rückseiten) auf Basis von Stempeluntergruppen. Die Bilder werden dabei vorverarbeitet (Circle Crop, Graustufen) und in einem strukturierten Format für Trainingszwecke gespeichert.

## Ordnerstruktur der Rohdaten

Die Rohbilder müssen vorab in folgendem Format abgelegt sein:

entirety/  
├── obv/   # Alle Vorderseiten-Bilder (nicht nach Klassen sortiert, einfach alle in einem Ordner)  
└── rev/   # Alle Rückseiten-Bilder (ebenfalls in einem Ordner)  

## Schritte zur Datenerstellung

### 1. `CreateTestData.py` ausführen

Wähle im Skript durch Setzen der Variablen `grayscale` und `cropped`, welche Vorverarbeitungsschritte du anwenden möchtest:

```python
grayscale = True
cropped = True
