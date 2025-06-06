# Keyword & Phrase Extractor (Polski)

Aplikacja webowa do wyciągania słów i fraz kluczowych z tekstów w języku polskim na podstawie TF-IDF.

## Funkcje:

✅ Ekstrakcja słów kluczowych (lematyzacja)  
✅ Ekstrakcja fraz kluczowych (noun chunks)  
✅ Wagi TF-IDF  
✅ Wykresy słupkowe  
✅ Eksport wyników do CSV  

## Uruchomienie lokalnie:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment na Streamlit Cloud:

1. Utwórz repozytorium na GitHub  
2. Dodaj `app.py`, `requirements.txt`, `README.md`  
3. Wejdź na [https://streamlit.io/cloud](https://streamlit.io/cloud)  
4. Kliknij "New app" → wybierz repozytorium → Deploy  

Aplikacja pojawi się pod adresem: `https://twoja-aplikacja.streamlit.app` 🚀
