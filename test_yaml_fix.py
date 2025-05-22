#!/usr/bin/env python3
"""
Test script pour valider le fichier adyolov5s_simple.yaml corrigé
"""

import sys
import torch
import yaml
from pathlib import Path

# Ajouter le path YOLOv5-Face
sys.path.insert(0, '.')

def test_yaml_syntax():
    """Test la syntaxe YAML"""
    print("🔍 Test de la syntaxe YAML...")
    
    yaml_path = 'models/adyolov5s_simple.yaml'
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   ✅ Syntaxe YAML valide")
        print(f"   Classes: {config['nc']}")
        print(f"   Anchors: {len(config['anchors'])} niveaux")
        print(f"   Backbone: {len(config['backbone'])} couches")
        print(f"   Head: {len(config['head'])} couches")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"   ❌ Erreur de syntaxe YAML: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_model_creation():
    """Test la création du modèle"""
    print("\n🏗️ Test de création du modèle...")
    
    try:
        from models.yolo import Model
        
        # Créer le modèle
        model = Model('models/adyolov5s_simple.yaml', ch=3, nc=1)
        print(f"   ✅ Modèle créé avec succès")
        
        # Vérifier les couches de détection
        detect_layers = [m for m in model.model if hasattr(m, 'stride')]
        if detect_layers:
            detect_layer = detect_layers[0]
            print(f"   Couches de détection: {detect_layer.nl}")
            print(f"   Strides: {detect_layer.stride.tolist() if hasattr(detect_layer, 'stride') else 'Non définis'}")
            
            if detect_layer.nl == 4:
                print("   ✅ ADYOLOv5-Face confirmé (4 niveaux de détection)")
            else:
                print(f"   ⚠️ Attendu 4 niveaux, trouvé {detect_layer.nl}")
        
        return True, model
        
    except Exception as e:
        print(f"   ❌ Erreur de création: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass(model):
    """Test du forward pass"""
    print("\n⚡ Test du forward pass...")
    
    try:
        model.eval()
        
        # Test avec différentes tailles d'image
        test_sizes = [640, 416, 320]
        
        for size in test_sizes:
            print(f"   Test avec image {size}x{size}...")
            
            test_input = torch.randn(1, 3, size, size)
            
            with torch.no_grad():
                output = model(test_input)
            
            if isinstance(output, tuple):
                pred, features = output
                print(f"     ✅ Output shape: {pred.shape}")
            else:
                print(f"     ✅ Output shape: {output.shape}")
        
        print("   ✅ Tous les forward pass réussis")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur de forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_mode(model):
    """Test du mode entraînement"""
    print("\n🎯 Test du mode entraînement...")
    
    try:
        model.train()
        
        test_input = torch.randn(1, 3, 640, 640)
        
        # Test forward en mode training
        output = model(test_input)
        
        if isinstance(output, list):
            print(f"   ✅ Mode training: {len(output)} sorties de feature maps")
            for i, out in enumerate(output):
                if out is not None:
                    print(f"     Niveau {i}: {out.shape}")
        else:
            print(f"   ✅ Mode training: Output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur en mode training: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Test de Validation ADYOLOv5-Face YAML")
    print("=" * 50)
    
    # Test 1: Syntaxe YAML
    if not test_yaml_syntax():
        print("\n❌ Échec du test de syntaxe YAML")
        return False
    
    # Test 2: Création du modèle
    success, model = test_model_creation()
    if not success:
        print("\n❌ Échec de création du modèle")
        return False
    
    # Test 3: Forward pass
    if not test_forward_pass(model):
        print("\n❌ Échec du forward pass")
        return False
    
    # Test 4: Mode entraînement
    if not test_training_mode(model):
        print("\n❌ Échec du mode entraînement")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 TOUS LES TESTS RÉUSSIS!")
    print("✅ Le fichier adyolov5s_simple.yaml est valide")
    print("✅ ADYOLOv5-Face peut être créé et utilisé")
    print("✅ Prêt pour l'entraînement sur Google Colab")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
