import streamlit as st
import fitz
import pandas as pd
import docx
from PIL import Image
import io
import pyttsx3
import multiprocessing
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sendgrid
from sendgrid.helpers.mail import Mail
import os
import base64
from cryptography.fernet import Fernet

key = os.getenv("ENCRYPTION_KEY")
if not key:
    raise EnvironmentError("ENCRYPTION_KEY not set in environment variables.")
cipher_suite = Fernet(key)

analyzer = SentimentIntensityAnalyzer()

def dynamic_code_generator(func_name, params, operation):
    try:
        code = f"""
def {func_name}({params}):
    result = {operation}
    return result
"""
        exec(code, globals())
    except Exception as e:
        st.error(f"Code gen error: {str(e)}")





def transform(data):
    try:
        encoded = base64.b64encode(data.encode())
        encrypted_data = cipher_suite.encrypt(encoded)
        return encrypted_data
    except Exception as e:
        st.error(f"Data obfuscation failed: {str(e)}")
        return None



def detransform(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        decoded = base64.b64decode(decrypted_data).decode()
        return decoded
    except Exception as e:
        st.error(f"Decryption error: {str(e)}")
        return None





def comp_hashing(text):
    return hashlib.sha256(text.encode()).hexdigest()


# Open and manage database connections
def create_db_connection(db_path=":memory:"):
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"DB connection failed: {str(e)}")
        return None





def store_embeddings(conn, doc_id, chunk_id, embedding):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT,
                chunk_id TEXT,
                embedding BLOB
            )
        """)
        cursor.execute("""
            INSERT INTO embeddings (doc_id, chunk_id, embedding)
            VALUES (?, ?, ?)
        """, (doc_id, chunk_id, embedding))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"DB insertion err: {str(e)}")



# def retrieve_embeddings(conn, doc_id):
#     try:
#         cursor = conn.cursor()
#         cursor.execute("SELECT chunk_id, embedding FROM embeddings WHERE doc_id = ?", (doc_id,))
#         rows = cursor.fetch()
#         return {row[0]: row[1] for row in rows}

#
#
#
# def parallel_query_processing(texts, query, model_name="gpt-4"):
#     try:
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = [executor.submit(single_query, text, query, model_name) for texts in texts]
#             return [future.result() for future in futures]




def retrieve_embeddings(conn, doc_id):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, embedding FROM embeddings WHERE doc_id = ?", (doc_id,))
        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}
    except sqlite3.Error as e:
        st.error(f"DB retrieval err: {str(e)}")
        return {}



def parallel_query_processing(texts, query, model_name="gpt-4"):
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(single_query, text, query, model_name) for text in texts]
            return [future.result() for future in futures]
    except Exception as e:
        st.error(f"Parallel query error: {str(e)}")
        return []







def single_query(text, query, model_name):
    try:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        conn = create_db_connection()

        if conn is None:
            return "DB connection failed."

        doc_id = comp_hashing(text)
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_text(chunk)
            transformd_embedding = transform(embedding)
            if transformd_embedding:
                store_embeddings(conn, doc_id, f"chunk_{i}", transformd_embedding)

        llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

        knowledge_base = retrieve_embeddings(conn, doc_id)
        retriever = DummyRetriever(knowledge_base, embeddings)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain.run(query)

        return detransform(response)
    except Exception as e:
        st.error(f"Query handling error: {str(e)}")
        return "Query failed."






class DummyRetriever:
    def __init__(self, knowledge_base, embeddings):
        self.knowledge_base = knowledge_base
        self.embeddings = embeddings

    def retrieve(self, query):
        return list(self.knowledge_base.values())


def llm_summary(text, summary_type="Overall"):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        messages = [
            {"role": "system", "content": "You are a great summarization assistant..."},
            {"role": "user", "content": f"Provide a {summary_type.lower()} summary of the following text:\n\n{text}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.1,
        )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Summary gen error: {str(e)}")
        return "Summary failed."





# Preprocess  count or TF-IDF
def preprocess_text(documents, method='count'):
    try:
        if method == 'count':
            vectorizer = CountVectorizer(stop_words='english')
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(documents)
        return dtm, vectorizer
    except Exception as e:
        st.error(f"Text preprocessing error: {str(e)}")
        return None, None


# Topic modeling (LDA, NMF, LSA)
def topic_modeling(dtm, model_type='LDA', num_topics=5):
    try:
        if model_type == 'LDA':
            model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        elif model_type == 'NMF':
            model = NMF(n_components=num_topics, random_state=42)
        elif model_type == 'LSA':
            model = TruncatedSVD(n_components=num_topics, random_state=42)
        model.fit(dtm)
        return model
    except Exception as e:
        st.error(f"Topic modeling error: {str(e)}")
        return None



def topic_words(model, vectorizer, num_words=10):
    try:
        words = np.array(vectorizer.get_feature_names_out())
        topics = []
        for topic in model.components_:
            top_words_idx = topic.argsort()[-num_words:]
            topics.append((words[top_words_idx], topic[top_words_idx]))
        return topics
    except Exception as e:
        st.error(f"Topic word extraction error: {str(e)}")
        return []



def plot_topic_words_bar_chart(topics, model_type):
    try:
        for i, (words, weights) in enumerate(topics):
            y_pos = np.arange(len(words))
            plt.figure(figsize=(10, 5))
            plt.barh(y_pos, weights, align='center', color='red')
            plt.yticks(y_pos, words)
            plt.xlabel('Weight')
            plt.title(f'{model_type} - Topic {i + 1}')
            plt.gca().invert_yaxis()
            st.pyplot(plt)
    except Exception as e:
        st.error(f"Bar chart plotting err: {str(e)}")


#heatmap of word distributions across topics
def plot_topic_words_heatmap(model, vectorizer, model_type):
    try:
        words = np.array(vectorizer.get_feature_names_out())
        topic_word_distributions = model.components_
        df = pd.DataFrame(data=topic_word_distributions, columns=words)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.T, annot=False, cmap="Reds", cbar=True)
        plt.title(f'{model_type} - Heatmap of Word Distributions Across Topics')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Heatmap plotting error: {str(e)}")


#PCA and TSNE
def plot_pca_tsne(model, dtm, model_type, method='PCA'):
    try:
        doc_topic_distributions = model.transform(dtm)
        if doc_topic_distributions.shape[0] < 2:
            st.write(f"Not enough samples for {method}. Skipping visualization.")
            return
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            title = f'{model_type} - PCA Plot of Topic Distributions'
        else:
            reducer = TSNE(n_components=2, random_state=42)
            title = f'{model_type} - TSNE Plot of Topic Distributions'
        reduced_topics = reducer.fit_transform(doc_topic_distributions)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_topics[:, 0], reduced_topics[:, 1], c='red')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"{method} plotting err: {str(e)}")





# Keyword frequency
def keyword_frequency_per_page(text_per_page, keywords):
    try:
        frequency_data = []
        for text in text_per_page:
            page_counts = []
            for keyword in keywords:
                count = text.lower().count(keyword.lower())
                page_counts.append(count)
            frequency_data.append(page_counts)
        return pd.DataFrame(frequency_data, columns=keywords,
                            index=[f"Page {i}" for i in range(1, len(text_per_page) + 1)])
    except Exception as e:
        st.error(f"Keyword freq err: {str(e)}")
        return pd.DataFrame()




#VADER
def vader_sentiment_per_page(text_per_page):
    try:
        sentiment_scores = []
        for text in text_per_page:
            score = analyzer.polarity_scores(text)
            sentiment_scores.append(score['compound'])  # Using the compound score for overall sentiment
        return sentiment_scores
    except Exception as e:
        st.error(f"Sentiment analysis err: {str(e)}")
        return []



#sentiment
def contributing_words_vader(text_per_page):
    try:
        sentiment_details = []
        for text in text_per_page:
            positive_words = []
            negative_words = []
            words = text.split()
            for word in words:
                score = analyzer.polarity_scores(word)['compound']
                if score > 0:
                    positive_words.append(word)
                elif score < 0:
                    negative_words.append(word)
            sentiment_details.append((positive_words, negative_words))
        return sentiment_details
    except Exception as e:
        st.error(f"Contributing words extraction err: {str(e)}")
        return []


# TTS
def tts_process(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS process err: {str(e)}")





st.title("Perkins&Will RFP Analysis Engine")



#=======================================================================



if "tts_process" not in st.session_state:
    st.session_state["tts_process"] = None

#custom CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; }
    h1 { color: #f50505; font-family: 'Orbitron', sans-serif; text-align: center; }
    .streamlit-expanderHeader { color: #f50505; font-size: 18px; font-weight: bold; }
    .css-1cpxqw2 { background-color: #28293e; border: 2px solid #f50505; color: #f50505; font-weight: bold; }
    .css-1n543e5 .stSlider { color: #f50505; }
    .css-1n543e5 .stSlider > div { background: #f50505; }
    .css-1n543e5 .stNumberInput input { background-color: #000000; color: #f50505; border: 1px solid #f50505; }
    .css-18e3th9 { background-color: #28293e; }
    button { background-color: #28293e; border: 2px solid #f50505; color: #f50505; font-weight: bold; }
    ::-webkit-scrollbar { width: 12px; }
    ::-webkit-scrollbar-track { background: #000000; }
    ::-webkit-scrollbar-thumb { background-color: #f50505; border-radius: 20px; border: 3px solid #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)


# #custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp { background-color: #000000; }
#     h1 { color: #f50505; font-family: 'Orbitron', sans-serif; text-align: center; }
#     .streamlit-expanderHeader { color: #f50505; font-size: 18px; font-weight: bold; }
#     .css-1cpxqw2 { background-color: #28293e; border: 2px solid #f50505; color: #f50505; font-weight: bold; }
#     .css-1n543e5 .stSlider { color: #f50505; }
#     .css-1n543e5 .stSlider > div { background: #f50505; }
#     .css-1n543e5 .stNumberInput input { background-color: #000000; color: #f50505; border: 1px solid #f50505; }
#     .css-18e3th9 { background-color: #28293e; }
#     button { background-color: #28293e; border: 2px solid #f50505; color: #f50505; font-weight: bold; }
#     ::-webkit-scrollbar { width: 12px; }
#     ::-webkit-scrollbar-track { background: #000000; }
#     ::-webkit-scrollbar-thumb { background-color: #f50505; border-radius: 20px; border: 3px solid #000000; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )



uploaded_files = st.file_uploader("Upload RFP files (.pdf, .xlsx, .xls, .docx, .doc)",
                                  type=["pdf", "xlsx", "xls", "docx", "doc"],
                                  accept_multiple_files=True)


if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            # Parse PDF
            try:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                num_pages = doc.page_count

                # Extract text
                text_per_page = [page.get_text() for page in doc]

                # Combine
                text = " ".join(text_per_page)
            except Exception as e:
                st.error(f"Error parsing PDF: {str(e)}")
                continue







            # overview Expander  3
            with st.expander(f"Overview : {uploaded_file.name}", expanded=True):

                tab1, tab2, tab3, tab4 = st.tabs(["PDF View", "Text Extraction", "Image Extraction", "Text to Speech"])

                with tab1:
                    try:
                        st.write(f"Total Pages: {num_pages}")
                        page_number = st.slider("Select a page to view", 1, num_pages, 1,
                                                key=f"slider_{uploaded_file.name}")
                        page = doc.load_page(page_number - 1)
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        st.image(img_data, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying PDF page: {str(e)}")




                    st.header("Summary")
                    page_text = text_per_page[page_number - 1]
                    if st.button("Generate Summary", key=f"summary_button_{uploaded_file.name}"):
                        summary = llm_summary(page_text)
                        st.write(f"**Summary of Page {page_number}:**")
                        st.write(summary)




                    st.header("RAG (Retrieval-Augmented Generation)")
                    user_query = st.text_input("Ask a question about this document:", key=f"query_{uploaded_file.name}")
                    if st.button("Run Query", key=f"query_button_{uploaded_file.name}"):
                        if user_query:
                            response = single_query(text, user_query, "gpt-4")
                            st.write("**Response:**")
                            st.write(response)
                        else:
                            st.warning("Please enter a query.")


                with tab2:
                    if st.button("Notify Stakeholders", key=f"notify_{uploaded_file.name}"):

                        keyword_to_recipients = {
                            "implementing software": [""],
                        }


                        for keyword, recipients in keyword_to_recipients.items():
                            if keyword.lower() in text.lower():
                                for recipient in recipients:
                                    send_email(
                                        subject="Important Document Notification",
                                        to_email=recipient,
                                        content=f"The document '{uploaded_file.name}' contains the phrase '{keyword}'. Please review it."
                                    )
                                st.success(f"Notification sent to {', '.join(recipients)} for keyword '{keyword}'!")

                    st.write(text)



                with tab3:
                    try:
                        images = []
                        for page in doc:
                            images.extend(page.get_images(full=True))
                        if images:
                            for img_index, img in enumerate(images):
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_data = base_image["image"]
                                image = Image.open(io.BytesIO(image_data))
                                st.image(image, caption=f"Image {img_index + 1}", use_column_width=True)


                        else:
                            st.write("No images found in the PDF document.")
                    except Exception as e:
                        st.error(f"Error extracting images: {str(e)}")

                with tab4:
                    if st.button("Start Text to Speech", key=f"start_button_{uploaded_file.name}"):

                        if st.session_state["tts_process"] is not None:
                            st.warning("Audio is already playing. Please stop it first.")
                        else:
                            st.session_state["tts_process"] = multiprocessing.Process(target=tts_process, args=(text,))

                            st.session_state["tts_process"].start()

                    if st.button("Stop Text to Speech", key=f"stop_button_{uploaded_file.name}"):


                        if st.session_state["tts_process"] is not None:
                            st.session_state["tts_process"].terminate()
                            st.session_state["tts_process"] = None
                            st.success("Audio stopped.")
                        else:
                            st.warning("No audio is currently playing.")


            with st.expander(f"Analysis : {uploaded_file.name}", expanded=True):
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Word Cloud", "Keyword Frequency Heat-Map", "Polarity Sentiment Analysis", "Topic Modeling"])

                with tab1:



                    custom_stopwords = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

                    if text.strip():
                        try:
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color="black",
                                colormap="Reds",
                                stopwords=STOPWORDS.union(custom_stopwords)
                            ).generate(text)


                            keywords = sorted(wordcloud.words_.keys(), key=lambda x: wordcloud.words_[x], reverse=True)[
                                       :10]

                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation="bilinear")
                            plt.axis("off")
                            st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                    else:
                        st.write("No text available for word cloud generation.")

                with tab2:
                    if text.strip():
                        if keywords:
                            try:

                                df = keyword_frequency_per_page(text_per_page, keywords)

                                # Dynamically adjust figure size
                                num_pages = len(text_per_page)
                                num_keywords = len(keywords)
                                plt.figure(figsize=(num_keywords * 2, num_pages * 0.6))


                                ax = sns.heatmap(
                                    df,
                                    annot=True,
                                    cmap="Reds",
                                    linewidths=0.5,
                                    linecolor='gray',
                                    cbar=True,
                                    square=False,
                                    fmt="d",
                                    robust=True
                                )

                                plt.title("Keyword Frequency Heatmap Per Page", fontsize=14)
                                plt.ylabel("Pages", fontsize=12)
                                plt.xlabel("Keywords", fontsize=12)
                                plt.xticks(rotation=45, ha='right', fontsize=10)

                                ax.set_yticks(range(len(df)))
                                ax.set_yticklabels(df.index, rotation=0, fontsize=10)

                                plt.tight_layout()
                                st.pyplot(plt)
                            except Exception as e:
                                st.error(f"Error generating heatmap: {str(e)}")

                        else:
                            st.write("No keywords available for heatmap generation.")


                    else:
                        st.write("No text available for keyword frequency heatmap generation.")

                with tab3:
                    if text.strip():
                        try:
                            sentiments = vader_sentiment_per_page(text_per_page)
                            plt.figure(figsize=(15, 8))
                            plt.plot(range(1, len(sentiments) + 1), sentiments, marker='o', linestyle='-', color='red')
                            plt.title("Sentiment Polarity Per Page")
                            plt.xlabel("Page Number")
                            plt.ylabel("Sentiment Polarity")
                            plt.axhline(0, color='black', linestyle='--')  # Neutral sentiment line

                            plt.xticks(range(1, len(sentiments) + 1), rotation=90)
                            plt.grid(True)
                            st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error generating sentiment analysis: {str(e)}")
                    else:
                        st.write("No text available for sentiment analysis.")

                with tab4:
                    all_pages = st.checkbox("All Pages", key=f"all_pages_{uploaded_file.name}")

                    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=5, step=1,
                                           key=f"num_topics_slider_{uploaded_file.name}")

                    if not all_pages:
                        page_number = st.slider("Select a page to analyze", 1, num_pages, 1,
                                                key=f"analyze_page_slider_{uploaded_file.name}")
                        selected_text = [text_per_page[page_number - 1]]
                    else:
                        selected_text = text_per_page

                    model_type = st.selectbox("Select Topic Modeling Method", ['LDA', 'NMF', 'LSA'],
                                              key=f"model_type_selector_{uploaded_file.name}")


                    dtm, vectorizer = preprocess_text(selected_text, method='tfidf')


                    model = topic_modeling(dtm, model_type=model_type, num_topics=num_topics)


                    topics = topic_words(model, vectorizer)
                    for i, (words, weights) in enumerate(topics):
                        st.write(f"**Topic {i + 1}:** {', '.join(words)}")

                    # Plot bar charts
                    plot_topic_words_bar_chart(topics, model_type)

                    # Plot heatmap f
                    plot_topic_words_heatmap(model, vectorizer, model_type)

                    # Show PCA and TSNE
                    if all_pages:
                        plot_pca_tsne(model, dtm, model_type, method='PCA')
                        plot_pca_tsne(model, dtm, model_type, method='TSNE')

        elif file_type in ["xlsx", "xls"]:
            # Read Excel file
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                continue

            # Overview Expander
            with st.expander(f"Overview : {uploaded_file.name}", expanded=True):
                st.write(f"**Data Preview for {uploaded_file.name}:**")
                st.dataframe(df)

        elif file_type in ["docx", "doc"]:
            # Parse DOCX/DOC
            try:
                doc = docx.Document(uploaded_file)
                text_per_page = [paragraph.text for paragraph in doc.paragraphs]
                text = "\n".join(text_per_page)
                images = []
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        img = rel.target_part.blob
                        images.append(img)
            except Exception as e:
                st.error(f"Error parsing DOCX/DOC: {str(e)}")
                continue

            # Overview Expander
            with st.expander(f"Overview : {uploaded_file.name}", expanded=True):

                tab1, tab2, tab3 = st.tabs(["Text Extraction", "Image Extraction", "Text to Speech"])

                with tab1:
                    st.write(text)

                with tab2:
                    try:
                        if images:
                            for img_index, img_data in enumerate(images):
                                image = Image.open(io.BytesIO(img_data))
                                st.image(image, caption=f"Image {img_index + 1}", use_column_width=True)
                        else:
                            st.write("No images found in the Word document.")
                    except Exception as e:
                        st.error(f"Error extracting images: {str(e)}")

                with tab3:
                    if st.button("Start Text to Speech", key=f"start_button_{uploaded_file.name}"):
                        if st.session_state["tts_process"] is not None:
                            st.warning("Audio is already playing. Please stop it first.")
                        else:
                            st.session_state["tts_process"] = multiprocessing.Process(target=tts_process, args=(text,))
                            st.session_state["tts_process"].start()

                    if st.button("Stop Text to Speech", key=f"stop_button_{uploaded_file.name}"):
                        if st.session_state["tts_process"] is not None:
                            st.session_state["tts_process"].terminate()


                            st.session_state["tts_process"] = None
                            st.success("Audio stopped.")
                        else:
                            st.warning("No audio is currently playing.")

            # Analysis Expander
            with st.expander(f"Analysis : {uploaded_file.name}", expanded=True):
                tab1, tab2, tab3 = st.tabs(
                    ["Word Cloud", "Keyword Frequency Heat-Map", "Polarity Sentiment Analysis"])

                with tab1:
                    if text.strip():
                        try:
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color="black",
                                colormap="Reds",
                                stopwords=STOPWORDS
                            ).generate(text)

                            # Extract top keywords
                            keywords = sorted(wordcloud.words_.keys(), key=lambda x: wordcloud.words_[x], reverse=True)[
                                       :10]

                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation="bilinear")
                            plt.axis("off")

                            st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                    else:
                        st.write("No text available for word cloud generation.")

                with tab2:
                    if text.strip():
                        if keywords:
                            try:

                                df = keyword_frequency_per_page(text_per_page, keywords)

                                # Dynamically adjust figure size based on the number of pages and keywords
                                num_pages = len(text_per_page)
                                num_keywords = len(keywords)
                                plt.figure(figsize=(num_keywords * 2, num_pages * 0.6))

                                # Create heatmap
                                ax = sns.heatmap(
                                    df,
                                    annot=True,
                                    cmap="Reds",
                                    linewidths=0.5,
                                    linecolor='gray',
                                    cbar=True,
                                    square=False,
                                    fmt="d",
                                    robust=True
                                )
                                plt.title("Keyword Frequency Heatmap Per Page", fontsize=14)

                                plt.ylabel("Pages", fontsize=12)
                                plt.xlabel("Keywords", fontsize=12)
                                plt.xticks(rotation=45, ha='right', fontsize=10)

                                ax.set_yticks(range(len(df)))
                                ax.set_yticklabels(df.index, rotation=0, fontsize=10)

                                plt.tight_layout()
                                st.pyplot(plt)
                            except Exception as e:
                                st.error(f"Error generating heatmap: {str(e)}")
                        else:
                            st.write("No keywords available for heatmap generation.")


                    else:
                        st.write("No text available for keyword frequency heatmap generation.")

                with tab3:
                    if text.strip():
                        try:
                            sentiments = vader_sentiment_per_page(text_per_page)


                            plt.figure(figsize=(15, 8))
                            plt.plot(range(1, len(sentiments) + 1), sentiments, marker='o', linestyle='-', color='red')

                            plt.title("Sentiment Polarity Per Page")

                            plt.xlabel("Page Number")
                            plt.ylabel("Sentiment Polarity")
                            plt.axhline(0, color='black', linestyle='--')  # Neutral sentiment line

                            plt.xticks(range(1, len(sentiments) + 1), rotation=90)
                            plt.grid(True)

                            st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error generating sentiment analysis: {str(e)}")
                    else:
                        st.write("No text available for sentiment analysis.")
