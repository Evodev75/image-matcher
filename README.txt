# Image Matcher API

Déploiement prêt pour Railway.

## Endpoints

- `POST /match` : upload d'une image (champ: `file`, type: `form-data`)
- Retourne : `{{ "match_id": "Valide_postif_AMP" }}`

## Structure attendue des fichiers de référence

Place tes 18 images dans `reference_images/` avec des noms de type :
- toda_ref__0000s_0001_Valide_postif_AMP.jpg
- toda_ref__0000s_0002_Negatif_T1.jpg
- etc.

L'ID retourné sera extrait entre `0000s_` et l'extension.

Exemple : `toda_ref__0000s_0004_Valide_postif_OPI.jpg` → `Valide_postif_OPI`

## Commande de lancement locale (optionnel)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
