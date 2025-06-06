import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd

# --- KeywordExtractorPremium ---
class KeywordExtractorPremium:
    def __init__(self, model_name="pl_core_news_sm"):
        self.nlp = spacy.load(model_name)
        self.stop_words = self.nlp.Defaults.stop_words
    
    def noun_chunks_extractor(self, text):
        doc = self.nlp(text)
        return [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) > 1
            and not any(word in self.stop_words for word in chunk.text.lower().split())
        ]
    
    def lemmatized_keywords_extractor(self, text):
        doc = self.nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {'NOUN', 'PROPN', 'ADJ'}
            and not token.is_stop
            and not token.is_punct
            and token.lemma_.lower() not in self.stop_words
        ]
    
    def compute_tfidf(self, texts, extractor, top_n=10):
        tokenized_texts = [" ".join(extractor(text)) for text in texts]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_array = tfidf_matrix.toarray()
        
        tfidf_scores = tfidf_array.sum(axis=0)
        scores = list(zip(feature_names, tfidf_scores))
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return scores_sorted[:top_n]
    
    def extract_keywords(self, texts, top_n=10):
        if isinstance(texts, str):
            texts = [texts]
        return self.compute_tfidf(texts, self.lemmatized_keywords_extractor, top_n)
    
    def extract_phrases(self, texts, top_n=10):
        if isinstance(texts, str):
            texts = [texts]
        return self.compute_tfidf(texts, self.noun_chunks_extractor, top_n)
    
    def plot_top_items(self, items, title="Top items"):
        labels, scores = zip(*items)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[::-1], scores[::-1])
        ax.set_title(title)
        ax.set_xlabel("TF-IDF Score")
        plt.tight_layout()
        return fig

# --- Streamlit app ---
st.set_page_config(page_title="Keyword & Phrase Extractor (Polski)", page_icon="📝")

st.title("📝 Keyword & Phrase Extractor (Polski) 🚀")
st.write("Wprowadź tekst poniżej, aby wyciągnąć słowa i frazy kluczowe na podstawie TF-IDF.")

input_text = st.text_area("Wpisz tekst tutaj:", height=300)

top_n = st.slider("Ile TOP fraz/słów pokazać?", min_value=5, max_value=20, value=10)

extractor = KeywordExtractorPremium()

if st.button("🔍 Wyciągnij słowa i frazy kluczowe"):
    if input_text.strip() == "":
        st.warning("❗️ Wprowadź tekst.")
    else:
        with st.spinner("Analizuję tekst..."):
            phrases = extractor.extract_phrases(input_text, top_n)
            keywords = extractor.extract_keywords(input_text, top_n)
        
        st.subheader("📌 Top Frazy Kluczowe")
        for phrase, score in phrases:
            st.write(f"**{phrase}** — {score:.4f}")
        
        st.pyplot(extractor.plot_top_items(phrases, title="Top Frazy Kluczowe"))
        
        st.subheader("📌 Top Słowa Kluczowe")
        for word, score in keywords:
            st.write(f"**{word}** — {score:.4f}")
        
        st.pyplot(extractor.plot_top_items(keywords, title="Top Słowa Kluczowe"))
        
        # --- Eksport do CSV ---
        df_phrases = pd.DataFrame(phrases, columns=["Fraza", "TF-IDF Score"])
        df_keywords = pd.DataFrame(keywords, columns=["Słowo", "TF-IDF Score"])
        
        st.download_button(
            label="⬇️ Pobierz frazy do CSV",
            data=df_phrases.to_csv(index=False).encode('utf-8'),
            file_name='frazy_kluczowe.csv',
            mime='text/csv'
        )
        
        st.download_button(
            label="⬇️ Pobierz słowa do CSV",
            data=df_keywords.to_csv(index=False).encode('utf-8'),
            file_name='slowa_kluczowe.csv',
            mime='text/csv'
        )
