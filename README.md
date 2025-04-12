# Sports Image Classification - Deep Learning Project

Ce projet Deep Learning permet d'entraîner un modèle de vision par ordinateur pour classer automatiquement des images sportives.  
Il repose sur un modèle **ResNet18 pré-entraîné** et un dataset open-source disponible sur Hugging Face.

---

## 🔗 Dataset

**Nom :** [HES-XPLAIN / SportsImageClassification](https://huggingface.co/datasets/HES-XPLAIN/SportsImageClassification)  
**Contenu :** Images réparties en 3 dossiers (`train`, `valid`, `test`) avec des sous-dossiers par classe.  
**Format :** Compatible avec `torchvision.datasets.ImageFolder`.

---

## 🚀 Étapes pour exécuter le projet

### 1. Cloner le dataset Hugging Face

Assurez-vous d’avoir Git LFS :

```bash
git lfs install

git clone https://huggingface.co/datasets/HES-XPLAIN/SportsImageClassification

```
Il faut obtenir la structure suivante:

SportsImageClassification/
├── train/
├── valid/
├── test/
├── idx_to_names.json
└── sports.csv

Ensuite créer un environemment python (venv)

Puis: pip install -r requirements.txt


Merci et bonne classification !

Morgan Jowitt Gaétan Dumas