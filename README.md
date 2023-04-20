# Whisper_demo

Voici un exemple de code Python pour extraire l'audio d'un fichier vidéo au format mp4, transcrire le contenu en utilisant le modèle Whisper d'OpenAI, puis intégrer les résultats de transcription avec le modèle de génération de langage CHATGPT-3 pour obtenir un résumé.

## Introduction

Whisper is a natural language processing model developed by OpenAI, also known as GPT-3 Whisper. The model was trained on a large corpus of data to understand and generate natural phrases and conversations. It was developed to generate high-quality and accurate text for applications such as automatic transcription, content generation, dialogue, machine translation, and more.

Whisper est un modèle de traitement automatique de la parole développé par OpenAI, qui est également connu sous le nom de GPT-3 Whisper. Ce modèle a été formé sur un grand corpus de données pour comprendre et générer des phrases et des conversations naturelles. Il a été développé pour générer du texte avec une grande qualité et une précision élevée pour les applications de transcription automatique, de génération de contenu, de dialogue, de traduction automatique, et plus encore.

## installation

<https://pypi.org/project/moviepy>
<https://github.com/openai/whisper>
<https://www.assemblyai.com/blog/how-to-run-openais-whisper-speech-recognition-model/>
<https://beta.openai.com/docs/api-reference?lang=python>

# Python libraries to install

```bash
pip3 install PyPDF2
```

## Word Error rate

### Définition

La "Word Error Rate" (taux d'erreur de mots) est un indicateur couramment utilisé pour évaluer la qualité de la transcription automatique de la parole. Elle mesure le pourcentage de mots qui ont été mal reconnus par rapport au nombre total de mots dans un enregistrement donné. Plus le taux d'erreur est faible, meilleure est la qualité de la transcription.

### Graph

<https://www.assemblyai.com/blog/content/images/2022/09/multilingual_wer.png>

## Spectrogramme Mel

 un Spectrogramme Mel est un type de représentation visuelle de la fréquence des sons dans un signal audio. Il utilise une échelle de fréquences basée sur l'oreille humaine (échelle Mel) plutôt que sur une échelle linéaire de fréquences. Cela permet de mieux visualiser les caractéristiques de la voix humaine et d'autres sons vocaux. Les spectrogrammes Mel sont souvent utilisés dans les systèmes de reconnaissance vocale et d'analyse de la parole.

## Transcription en temps réel

 Possible, à condition de faire notre API et de le faire avec de chunks de 1 seconde.
