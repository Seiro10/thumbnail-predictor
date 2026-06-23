"""
Auto-labelling des thumbnails avec Google Cloud Vision API
Coût: ~$6-8 pour 1243 images
Temps: ~10-15 minutes
"""
import os
import pandas as pd
from google.cloud import vision
from tqdm import tqdm
import json
from pathlib import Path

# Configuration
# Uses Application Default Credentials (ADC) - run 'gcloud auth application-default login' first

# Initialiser le client Vision API
print("🔧 Initialisation de Google Vision API...")
client = vision.ImageAnnotatorClient()

# Charger le dataset
print("📊 Chargement du dataset...")
df = pd.read_csv('data/videos.csv')
print(f"✅ {len(df)} vidéos chargées")

# Fonction pour extraire l'expression dominante
def get_dominant_expression(face):
    """Détermine l'expression dominante d'un visage"""
    emotions = {
        'joy': face.joy_likelihood,
        'sorrow': face.sorrow_likelihood,
        'anger': face.anger_likelihood,
        'surprise': face.surprise_likelihood
    }

    # Convertir les enums en scores (0-5)
    emotion_scores = {}
    for emotion, likelihood in emotions.items():
        if likelihood == vision.Likelihood.VERY_LIKELY:
            emotion_scores[emotion] = 5
        elif likelihood == vision.Likelihood.LIKELY:
            emotion_scores[emotion] = 4
        elif likelihood == vision.Likelihood.POSSIBLE:
            emotion_scores[emotion] = 3
        elif likelihood == vision.Likelihood.UNLIKELY:
            emotion_scores[emotion] = 2
        else:  # VERY_UNLIKELY or UNKNOWN
            emotion_scores[emotion] = 1

    # Trouver l'émotion dominante
    dominant = max(emotion_scores.items(), key=lambda x: x[1])

    # Mapper vers nos catégories
    if dominant[1] >= 4:  # LIKELY ou VERY_LIKELY
        if dominant[0] == 'joy':
            return 'Sourire'
        elif dominant[0] == 'surprise':
            return 'Surprise'
        elif dominant[0] in ['anger', 'sorrow']:
            return 'Intense'

    return 'Neutre'

def get_color_category(dominant_colors):
    """Détermine si les couleurs sont chaudes, froides ou neutres"""
    if not dominant_colors.colors:
        return 'Neutre'

    # Prendre la couleur la plus dominante
    main_color = dominant_colors.colors[0].color

    r = main_color.red
    g = main_color.green
    b = main_color.blue

    # Couleurs chaudes: rouge/orange/jaune dominant
    if r > g and r > b and r > 150:
        return 'Chaud'

    # Couleurs froides: bleu/vert dominant
    if (b > r and b > g and b > 150) or (g > r and g > b and g > 150):
        return 'Froid'

    return 'Neutre'

def calculate_contrast(image_props):
    """Calcule le niveau de contraste approximatif"""
    if not image_props.dominant_colors.colors:
        return 'Moyen'

    # Analyser la distribution des couleurs
    colors = image_props.dominant_colors.colors

    # Si beaucoup de couleurs différentes = contraste élevé
    if len(colors) > 8:
        return 'Élevé'
    elif len(colors) < 4:
        return 'Faible'

    return 'Moyen'

def analyze_background(labels):
    """Analyse le fond basé sur les labels détectés"""
    label_texts = [label.description.lower() for label in labels]

    # Fond uni: peu de labels variés
    simple_backgrounds = ['sky', 'wall', 'ceiling', 'floor', 'plain']
    if any(bg in ' '.join(label_texts) for bg in simple_backgrounds):
        return 'Uni'

    # Fond chargé: beaucoup d'objets/labels
    if len(labels) > 10:
        return 'Chargé'

    # Flou: si "blur" ou "bokeh" détecté (rare)
    if 'blur' in ' '.join(label_texts) or 'bokeh' in ' '.join(label_texts):
        return 'Flou'

    # Par défaut, basé sur le nombre de labels
    if len(labels) > 7:
        return 'Chargé'
    else:
        return 'Uni'

# Analyser toutes les images
print("\n🖼️  Analyse des thumbnails avec Vision API...")
print("⏱️  Temps estimé: 10-15 minutes\n")

results = []
errors = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(row['thumbnail_path']):
            errors.append({'video_id': row['video_id'], 'error': 'File not found'})
            continue

        # Charger l'image
        with open(row['thumbnail_path'], 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # 1. Détection de visages
        faces_response = client.face_detection(image=image)
        faces = faces_response.face_annotations

        # 2. Détection de texte
        text_response = client.text_detection(image=image)
        texts = text_response.text_annotations

        # 3. Propriétés de l'image (couleurs)
        props_response = client.image_properties(image=image)
        props = props_response.image_properties_annotation

        # 4. Label detection (objets/scènes)
        labels_response = client.label_detection(image=image)
        labels = labels_response.label_annotations

        # Extraire les features
        nb_faces = len(faces)

        result = {
            'video_id': row['video_id'],

            # Visages
            'visage_present': nb_faces > 0,
            'nb_personnes': min(nb_faces, 2) if nb_faces <= 2 else 3,  # 0, 1, 2, ou 2+
            'expression': get_dominant_expression(faces[0]) if faces else 'Aucun',

            # Texte
            'texte_present': len(texts) > 1,  # Plus de 1 = du vrai texte (le 1er est toujours le texte complet)
            'texte_contenu': texts[0].description[:100] if texts else '',  # Premiers 100 caractères

            # Couleurs
            'couleur_dominante': get_color_category(props.dominant_colors),

            # Contraste (approximatif)
            'contraste': calculate_contrast(props),

            # Fond (basé sur les labels)
            'fond': analyze_background(labels),

            # Labels détectés (top 5)
            'labels': ','.join([label.description for label in labels[:5]]),

            # Scores de confiance
            'face_confidence': faces[0].detection_confidence if faces else 0,
            'text_confidence': texts[0].confidence if texts and len(texts) > 1 else 0,
        }

        results.append(result)

    except Exception as e:
        errors.append({
            'video_id': row['video_id'],
            'error': str(e)
        })
        continue

# Sauvegarder les résultats
print("\n💾 Sauvegarde des résultats...")

labels_df = pd.DataFrame(results)
labels_df.to_csv('data/auto_labels.csv', index=False)

print(f"\n✅ Analyse terminée!")
print(f"   ✓ {len(results)} images analysées avec succès")
print(f"   ✗ {len(errors)} erreurs")
print(f"\n📁 Fichier sauvegardé: data/auto_labels.csv")

# Afficher quelques statistiques
print("\n" + "="*70)
print("📊 STATISTIQUES DES LABELS")
print("="*70)

print("\n🧑 Visages:")
print(f"   Présents: {labels_df['visage_present'].sum()} ({labels_df['visage_present'].sum()/len(labels_df)*100:.1f}%)")
print(f"   Distribution nb_personnes:")
print(labels_df['nb_personnes'].value_counts().sort_index())

print("\n😊 Expressions:")
print(labels_df['expression'].value_counts())

print("\n📝 Texte:")
print(f"   Présent: {labels_df['texte_present'].sum()} ({labels_df['texte_present'].sum()/len(labels_df)*100:.1f}%)")

print("\n🎨 Couleurs dominantes:")
print(labels_df['couleur_dominante'].value_counts())

print("\n🌈 Contraste:")
print(labels_df['contraste'].value_counts())

print("\n🖼️  Fond:")
print(labels_df['fond'].value_counts())

# Sauvegarder les erreurs si il y en a
if errors:
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv('data/labelling_errors.csv', index=False)
    print(f"\n⚠️  {len(errors)} erreurs sauvegardées dans data/labelling_errors.csv")

print("\n" + "="*70)
print("🎯 PROCHAINE ÉTAPE")
print("="*70)
print("\nFusionner les labels avec le dataset principal:")
print("   python merge_labels.py")
print("\nOu continuez dans Jupyter pour explorer les labels!")


