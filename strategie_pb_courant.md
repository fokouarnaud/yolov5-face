# Stratégie d'implémentation ADYOLOv5-Face

*Date: 21 mai 2025 - Mise à jour: 22 mai 2025*

## Architecture ADYOLOv5-Face

ADYOLOv5-Face est une amélioration de YOLOv5-Face qui ajoute:
1. Un mécanisme Gather-and-Distribute (GD) à deux niveaux
2. Une tête de détection P2/4 pour les petits visages
3. Des modules de fusion d'attention et de transformer

### Structure des caractéristiques:
- P2/4: Petits visages (taille 4)
- P3/8: Visages moyens (taille 8) 
- P4/16: Visages normaux (taille 16)
- P5/32: Grands visages (taille 32)

## Approche d'implémentation

### 1. Modifications du fichier YAML (models/adyolov5s_simple.yaml)
- Structure principale basée sur YOLOv5s
- Ajout du backbone standard
- Ajout du FPN (Feature Pyramid Network) modifié avec GD
- Ajout de la couche P2/4 pour les petits visages
- Configuration de 4 têtes de détection distinctes

### 2. Modules GD (models/gd.py)
- `GDFusion`: Module principal pour la fusion des caractéristiques
- `AttentionFusion`: Pour la fusion de bas niveau (Low-Stage GD)
- `TransformerFusion`: Pour la fusion de haut niveau (High-Stage GD)

### 3. Compatibilité PyTorch 2.6+ (models/yolo.py)
- Fonction `normalize_args` améliorée
- Gestion des dimensions dynamiques
- Support des arguments sous forme de liste

### 4. Support multi-têtes (models/yolo.py)
- Modification de la classe `Model` pour détecter plusieurs couches `Detect`
- Gestion des strides multiples (4, 8, 16, 32)
- Fusion des sorties de détection de toutes les têtes

## Problèmes résolus

### 1. Erreur "list indices must be integers or slices, not list"
- **Cause**: Incompatibilité dans le parsing du YAML
- **Solution**: Implémenté une meilleure fonction `normalize_args()` pour traiter correctement les listes et les tuples
- **Fix**: Formaté les arguments YAML de manière cohérente dans adyolov5s_simple.yaml

### 2. Erreur d'importation circulaire avec Conv
- **Cause**: models/common.py importe depuis models/gd.py qui à son tour importe Conv depuis models/common.py
- **Solution**: Définir une version autonome de Conv directement dans gd.py au lieu de l'importer
- **Fix**: Copier l'implémentation de Conv dans gd.py pour briser la dépendance circulaire

## Problèmes potentiels et solutions

### 1. Problèmes avec les dimensions de tenseurs
- **Cause**: Incompatibilité entre dimensions lors de la fusion
- **Solution**: Vérifier les dimensions avec `print(x.shape)` aux endroits critiques
- **Fix**: Adapter dynamiquement les dimensions ou ajouter des couches d'adaptation

### 2. Erreurs de mémoire CUDA
- **Cause**: Architecture plus complexe requérant plus de mémoire
- **Solution**: Réduire la taille de batch ou utiliser l'accumulation de gradient
- **Fix**: Utiliser `--batch-size 8` ou `--accumulate 2` lors de l'entraînement

### 3. Problèmes avec les anchors
- **Cause**: Anchors non adaptés aux petits visages
- **Solution**: Vérifier la répartition des anchors, notamment pour la tête P2/4
- **Fix**: Recalculer les anchors avec `--gen-anchors` sur votre dataset

## Importations circulaires – Bonnes pratiques

Pour éviter les importations circulaires dans le projet:

1. **Conception modulaire**:
   - Chaque module doit être indépendant et avoir des responsabilités clairement définies
   - Éviter les dépendances mutuelles entre modules

2. **Solutions pour les importations circulaires**:
   - **Solution 1**: Définir les classes utilitaires comme Conv directement dans chaque module qui en a besoin
   - **Solution 2**: Déplacer les classes communes dans un module de base qui ne dépend d'aucun autre
   - **Solution 3**: Utiliser des imports conditionnels (import à l'intérieur des fonctions)

3. **Structure recommandée**:
   ```
   models/
   ├── base.py        # Classes de base comme Conv, sans dépendances
   ├── common.py      # Modules communs qui peuvent dépendre de base.py
   ├── gd.py          # Mécanisme GD qui peut dépendre de base.py
   └── yolo.py        # Module principal qui peut dépendre de tout
   ```

## Points de débogage

En cas d'erreur lors de l'initialisation du modèle:
1. Activer la fonction `debug_parse()` dans `parse_model()`
2. Ajouter des prints temporaires pour afficher les dimensions à chaque étape
3. Vérifier que chaque module reçoit et produit des tenseurs aux bonnes dimensions

En cas d'erreur pendant l'entraînement:
1. Vérifier la propagation des gradients avec `print(x.requires_grad)` 
2. Surveiller les valeurs des tenseurs avec `print(torch.isnan(x).any())`
3. Ajouter le flag `--debug` pour obtenir plus d'informations

## Prochaines étapes

1. **Implémentation de Conv dans gd.py**
   - Copier l'implémentation de Conv depuis models/common.py
   - Adapter le code pour éviter l'importation circulaire
   - Tester le modèle pour vérifier que l'erreur est résolue

2. **Tests et validation**
   - Tester le modèle avec un petit dataset
   - Vérifier que toutes les têtes de détection fonctionnent correctement
   - Comparer les performances avec YOLOv5-Face standard

3. **Optimisations futures**
   - Ajuster les hyperparamètres pour améliorer la détection des petits visages
   - Implémenter des techniques de data augmentation spécifiques
   - Explorer d'autres variantes du mécanisme GD pour les petits visages
