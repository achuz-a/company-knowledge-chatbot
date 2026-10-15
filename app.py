
import streamlit as st

from rag import (
    answer_question,
    get_collection,
    index_documents,
    retrieve_chunks,
)


st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Company Knowledge Assistant")
st.write(
    "Ask questions about your company documents "
    "and receive answers with source references."
)

st.info(
    "Upload and index documents before asking questions. "
    "Only upload documents you are authorized to process."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("Document library")

    uploaded_files = st.file_uploader(
        "Upload company documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT and DOCX.",
    )

    st.caption(
        "Indexing new files replaces the existing index "
        "in this starter version."
    )

    if st.button(
        "Index uploaded documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        documents = [
            {
                "name": file.name,
                "data": file.getvalue(),
            }
            for file in uploaded_files
        ]

        try:
            with st.spinner(
                "Extracting text, generating embeddings, "
                "and building the search index..."
            ):
                result = index_documents(documents)

            st.session_state.indexed_files = [
                file.name for file in uploaded_files
            ]
            st.session_state.messages = []

            st.success(
                f"Indexed {result['document_count']} document(s) "
                f"into {result['chunk_count']} searchable chunks."
            )

        except Exception as error:
            st.error(
                f"Indexing failed: {error}"
            )

    collection = get_collection()

    if collection is not None:
        st.success(
            f"Knowledge base ready: {collection.count()} chunks"
        )
    else:
        st.warning("No indexed documents found.")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.subheader("Chat")

# Display existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        page = source["page"]

                        location = (
                            f"Page {page}"
                            if page
                            else "Page not available"
                        )

                        st.markdown(
                            f"**{source['reference']} — "
                            f"{source['source']} ({location})**"
                        )
                        st.write(source["text"])

question = st.chat_input(
    "Ask a question about your company documents..."
)

if question:
    collection = get_collection()

    if collection is None:
        st.warning(
            "Please upload and index documents before asking questions."
        )

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching documents and preparing an answer..."
                ):
                    sources = retrieve_chunks(
                        question,
                        top_k=5,
                    )

                    answer = answer_question(
                        question,
                        sources,
                    )

                st.markdown(answer)

                with st.expander("View retrieved sources"):
                    for source in sources:
                        page = source["page"]

                        location = (
                            f"Page {page}"
                            if page
                            else "Page not available"
                        )

                        st.markdown(
                            f"**{source['reference']} — "
                            f"{source['source']} ({location})**"
                        )
                        st.write(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(
                    f"Could not answer this question: {error}"
                )