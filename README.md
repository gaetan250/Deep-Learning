# Sports Image Classifier with Grad-CAM

Ce projet Deep Learning permet d'entraîner un modèle de vision par ordinateur pour classer automatiquement des images sportives.
Il repose sur un modèle ResNet18 pré-entraîné et un dataset open-source disponible sur Hugging Face.

---

## 🧠 Objectif du projet

Ce projet vise à identifier le sport représenté sur une image parmi **100 catégories** (football, ski, surf, etc.) grâce à un modèle d’apprentissage profond. Il inclut :
- L'entraînement de plusieurs architectures CNN sur un dataset open-source,
- La visualisation des performances (matrice de confusion, erreurs),
- Une interface web interactive avec **Grad-CAM** pour comprendre les décisions du modèle.

---


## 🔗 Dataset

**Nom :** [HES-XPLAIN / SportsImageClassification](https://huggingface.co/datasets/HES-XPLAIN/SportsImageClassification)  
**Contenu :** Images réparties en 3 dossiers (`train`, `valid`, `test`) avec des sous-dossiers par classe.  
**Format :** Compatible avec `torchvision.datasets.ImageFolder`.

---

## 📁 Structure du projet

Deep-Learning/
├── model/                       
│   ├── efficientnet_b0_.pth
│   └── vgg16-.pth
│
├── SportsImageClassification/  
│   ├── train/                    
│   ├── valid/                    
│   └── test/                     
├── app.py                       
├── deep.ipynb                  
├── class_names.pkl
├── resnet_model.pth                 
├── requirements.txt              
└── README.md    


## 🚀 Étapes pour exécuter le projet

### 1. Cloner le dataset Hugging Face

Assurez-vous d’avoir Git LFS :

```bash
git lfs install
git clone https://huggingface.co/datasets/HES-XPLAIN/SportsImageClassification  
```

Nous vous conseillons de créer un environnement virtuelle pui d'y installer les différentes dépendances.

```bash
pip install -r requirements.txt
```