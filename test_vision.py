"""
Script de test pour vérifier que Google Vision API fonctionne
"""
import os
from google.cloud import vision

try:
    print("🔧 Test de connexion à Google Vision API...\n")

    # Vérifier les credentials
    creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds:
        print(f"✅ Credentials trouvées: {creds}")
    else:
        print("⚠️  Variable GOOGLE_APPLICATION_CREDENTIALS non définie")
        print("   Tentative avec Application Default Credentials...")

    # Initialiser le client
    client = vision.ImageAnnotatorClient()
    print("✅ Client Vision API initialisé\n")

    # Test sur une image du dataset
    import pandas as pd
    df = pd.read_csv('data/videos.csv')

    # Prendre la première image
    test_image_path = df.iloc[0]['thumbnail_path']

    if not os.path.exists(test_image_path):
        print(f"❌ Image de test introuvable: {test_image_path}")
        exit(1)

    print(f"🖼️  Test sur: {test_image_path}\n")

    # Charger et analyser l'image
    with open(test_image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Test de détection de visages
    response = client.face_detection(image=image)
    faces = response.face_annotations

    print(f"✅ Détection de visages: {len(faces)} visage(s) détecté(s)")

    # Test de détection de texte
    response = client.text_detection(image=image)
    texts = response.text_annotations

    print(f"✅ Détection de texte: {'Texte trouvé' if len(texts) > 1 else 'Pas de texte'}")

    # Test de détection de labels
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print(f"✅ Labels détectés: {len(labels)}")
    if labels:
        print(f"   Top 3: {', '.join([l.description for l in labels[:3]])}")

    print("\n" + "="*60)
    print("🎉 Vision API fonctionne parfaitement!")
    print("="*60)
    print("\n✅ Vous pouvez maintenant lancer:")
    print("   python auto_label_vision.py")

except Exception as e:
    print("\n" + "="*60)
    print("❌ ERREUR")
    print("="*60)
    print(f"\n{str(e)}\n")

    print("📋 Solutions possibles:\n")
    print("1. Vérifiez que Vision API est activée:")
    print("   https://console.cloud.google.com/apis/library/vision.googleapis.com")
    print("\n2. Configurez les credentials:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/credentials.json'")
    print("   Ou: gcloud auth application-default login")
    print("\n3. Installez la bibliothèque:")
    print("   pip install google-cloud-vision")
    print("\nVoir SETUP_VISION_API.md pour plus de détails")


