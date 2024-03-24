"""
Referências:
- [Streamlit Named Entity Recognition (NER) Annotation component](https://github.com/prasadchandan/st_ner_annotate)
- [Vs code debug](https://discuss.streamlit.io/t/vs-code-debug/520/7)
"""

from datetime import datetime

import pandas as pd
import streamlit as st

SENTIMENT_DICT = {"Positive": 1, "Negative": -1, "Neutral": 0}

CUSTOM_CSS = """
<style>
[data-testid="stApp"] {
  // background-color:red;
}

div[data-testid="stHorizontalBlock"] {
  // background-color: yellow;
  align-items: end;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]{
  // background-color: blue;
  display: flex;
  justify-content: end;
}
</style>
"""


def go_to_page():
    st.session_state["index"] = st.session_state["go_to_page_input"]


if __name__ == "__main__":
    st.markdown(
        CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.title("Sentiment Annotation Tool")

    uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

    if "index" not in st.session_state:
        st.session_state["index"] = 0

    if uploaded_file is not None:
        if "reviews_df" not in st.session_state:
            reviews_df = pd.read_csv(uploaded_file)
            st.session_state["reviews_df"] = reviews_df

        reviews_df = st.session_state.get("reviews_df")

        col_index, _, col_download_file = st.columns(3)
        with col_index:
            st.write("Index: " + str(st.session_state["index"]))

        with col_download_file:
            if st.button("Download file"):
                file_name, file_extension = uploaded_file.name.split(".")
                date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

                reviews_df = st.session_state.get("reviews_df")
                reviews_df.to_csv(
                    f"reviewed_{file_name}_{date_time_str}.{file_extension}",
                    index=False,
                )

        st.number_input(
            label="Go to Index:",
            step=1,
            value=st.session_state["index"],
            min_value=0,
            max_value=reviews_df.shape[0] - 1,
            key="go_to_page_input",
            on_change=go_to_page,
        )

        col_new_label, col_update_label = st.columns(2)
        with col_new_label:
            new_label = st.radio(
                label="New Label",
                options=list(SENTIMENT_DICT.keys()),
                key="new_label",
                horizontal=True,
                label_visibility="hidden",
            )

        with col_update_label:
            if st.button("Update Label", key="update_label"):
                reviews_df.at[st.session_state["index"], "sentiment"] = SENTIMENT_DICT[
                    new_label
                ]
                reviews_df.at[st.session_state["index"], "reviewed"] = 1

                st.session_state["index"] += 1
                st.session_state["reviews_df"] = reviews_df

        with st.container(border=True, height=240):
            st.write("#### Review Text")
            st.write(f"{reviews_df.iloc[st.session_state['index']]['review_text']}")

        col_current_label, col_predicted_label, col_predicted_score = st.columns(3)
        with col_current_label:
            list_sentiment_dict_keys = list(SENTIMENT_DICT.keys())
            list_sentiment_dict_values = list(SENTIMENT_DICT.values())

            sentiment = reviews_df.iloc[st.session_state["index"]]["sentiment"]
            sentiment_position = list_sentiment_dict_values.index(sentiment)

            st.text(f"Current Label: {list_sentiment_dict_keys[sentiment_position]}")

        with col_predicted_label:
            st.text(
                f"Predicted Label: {reviews_df.iloc[st.session_state['index']]['predicted_label']}"
            )

        with col_predicted_score:
            st.text(
                f"Predicted Score: {reviews_df.iloc[st.session_state['index']]['predicted_score']:.4f}"
            )
