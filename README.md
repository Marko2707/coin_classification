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
