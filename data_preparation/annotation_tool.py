"""
Referências:
- [Streamlit Named Entity Recognition (NER) Annotation component](https://github.com/prasadchandan/st_ner_annotate)
- [Vs code debug](https://discuss.streamlit.io/t/vs-code-debug/520/7)
"""

from datetime import datetime

import pandas as pd
import streamlit as st

SENTIMENT_DICT = {"Positive": 1, "Negative": -1, "Neutral": 0}


if __name__ == "__main__":
    st.title("Sentiment Annotation Tool")

    uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

    if "index" not in st.session_state:
        st.session_state["index"] = 0

    if uploaded_file is not None:
        # st.write(uploaded_file)

        if st.button("Download file"):
            file_name, file_extension = uploaded_file.name.split(".")
            date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

            reviews_df = st.session_state.get("reviews_df")
            reviews_df.to_csv(
                f"reviewed_{file_name}_{date_time_str}.{file_extension}",
                index=False,
            )

        if "reviews_df" not in st.session_state:
            reviews_df = pd.read_csv(uploaded_file)
            st.session_state["reviews_df"] = reviews_df

        reviews_df = st.session_state.get("reviews_df")

        col_predicted_label, col_predicted_score = st.columns(2)
        with col_predicted_label:
            st.text(
                f"Predicted Label: {reviews_df.iloc[st.session_state['index']]['predicted_label']}"
            )

        with col_predicted_score:
            st.text(
                f"Predicted Score: {reviews_df.iloc[st.session_state['index']]['predicted_score']}"
            )

        new_label = st.selectbox(
            label="Change Label",
            options=list(SENTIMENT_DICT.keys()),
            key="label",
        )

        col_previous, col_update, col_next = st.columns(3)
        with col_previous:
            if st.button(
                label="Previous",
                # disabled=st.session_state["index"] == 0,
                key="previous",
            ):
                if st.session_state["index"] > 0:
                    st.session_state["index"] -= 1

        with col_update:
            if st.button("Update", key="update"):
                reviews_df.at[st.session_state["index"], "sentiment"] = SENTIMENT_DICT[
                    new_label
                ]
                reviews_df.at[st.session_state["index"], "reviewed"] = 1

                st.session_state["index"] += 1
                st.session_state["reviews_df"] = reviews_df

        with col_next:
            if st.button(
                "Next",
                # disabled=st.session_state["index"] == reviews_df.shape[0],
                key="next",
            ):
                if st.session_state["index"] < reviews_df.shape[0]:
                    st.session_state["index"] += 1

        with st.container(border=True, height=320):
            st.write("#### Review Text")
            st.write(f"{reviews_df.iloc[st.session_state['index']]['review_text']}")

        col_index, col_reviewed = st.columns(2)
        with col_index:
            st.write("Index: " + str(st.session_state["index"]))

        with col_reviewed:
            if "reviewed" in reviews_df:
                st.write(
                    "Index: "
                    + str(reviews_df.at[st.session_state["index"], "reviewed"])
                )
