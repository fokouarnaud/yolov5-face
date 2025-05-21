# Stratégie d'implémentation ADYOLOv5-Face

*Date: 21 mai 2025*

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

## Problèmes potentiels et solutions

### 1. Erreur "list indices must be integers or slices, not list"
- **Cause**: Incompatibilité dans le parsing du YAML
- **Solution**: Utiliser la fonction de débogage `debug_parse()` pour identifier la couche problématique
- **Fix**: Corriger la structure des arguments dans le YAML ou adapter `normalize_args()`

### 2. Problèmes avec les dimensions de tenseurs
- **Cause**: Incompatibilité entre dimensions lors de la fusion
- **Solution**: Vérifier les dimensions avec `print(x.shape)` aux endroits critiques
- **Fix**: Adapter dynamiquement les dimensions ou ajouter des couches d'adaptation

### 3. Erreurs de mémoire CUDA
- **Cause**: Architecture plus complexe requérant plus de mémoire
- **Solution**: Réduire la taille de batch ou utiliser l'accumulation de gradient
- **Fix**: Utiliser `--batch-size 8` ou `--accumulate 2` lors de l'entraînement

### 4. Problèmes avec les anchors
- **Cause**: Anchors non adaptés aux petits visages
- **Solution**: Vérifier la répartition des anchors, notamment pour la tête P2/4
- **Fix**: Recalculer les anchors avec `--gen-anchors` sur votre dataset

## Points de débogage

En cas d'erreur lors de l'initialisation du modèle:
1. Activer la fonction `debug_parse()` dans `parse_model()`
2. Ajouter des prints temporaires pour afficher les dimensions à chaque étape
3. Vérifier que chaque module reçoit et produit des tenseurs aux bonnes dimensions

En cas d'erreur pendant l'entraînement:
1. Vérifier la propagation des gradients avec `print(x.requires_grad)` 
2. Surveiller les valeurs des tenseurs avec `print(torch.isnan(x).any())`
3. Ajouter le flag `--debug` pour obtenir plus d'informations

## Optimisations futures possibles

1. **Optimisation des hyperparamètres**:
   - Ajuster `small_face_weight` dans `hyp.adyolo.yaml`
   - Tester différentes valeurs pour `p2_weight`

2. **Améliorations du mécanisme GD**:
   - Expérimenter avec différentes architectures de TransformerFusion
   - Implémenter la version complète avec gating network

3. **Optimisations de performance**:
   - Ajouter du pruning pour réduire la taille du modèle
   - Implémenter la quantification pour accélérer l'inférence

4. **Adaptation des données**:
   - Augmenter la proportion de petits visages dans le dataset
   - Utiliser un sampling stratifié basé sur la taille des visages

## Notes sur la compatibilité avec Google Colab

- Utiliser torch>=2.0.0 pour la compatibilité avec la nouvelle implémentation
- Surveiller l'utilisation GPU avec `nvidia-smi` pendant l'entraînement
- Sauvegarder régulièrement les checkpoints en cas de déconnexion
- Utiliser TensorBoard pour surveiller la convergence des différentes têtes de détection
