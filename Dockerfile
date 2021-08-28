FROM nvcr.io/nvidia/mxnet:19.12-py3

RUN pip install lm_scorer==0.4.2 && \
    pip install git+git://github.com/ruanchaves/mlm-scoring.git@master#egg=mlm && \
    pip install emoji==1.4.1 python-json-logger==2.0.2 && \
    pip install -U spacy==3.1.0 && \
    ###
    # Large spaCy tokenizers
    ###
    # python -m spacy download ca_core_news_trf && \ 
    # python -m spacy download zh_core_web_trf && \
    # python -m spacy download da_core_news_trf && \
    # python -m spacy download nl_core_news_lg && \
    python -m spacy download en_core_web_trf && \
    # python -m spacy download fr_dep_news_trf && \
    # python -m spacy download de_dep_news_trf && \
    # python -m spacy download el_core_news_lg && \
    # python -m spacy download it_core_news_lg && \
    # python -m spacy download ja_core_news_lg && \
    # python -m spacy download lt_core_news_lg && \
    # python -m spacy download mk_core_news_lg && \
    # python -m spacy download xx_sent_ud_sm && \
    # python -m spacy download nb_core_news_lg && \
    # python -m spacy download pl_core_news_lg && \
    # python -m spacy download pt_core_news_lg && \
    # python -m spacy download ro_core_news_lg && \
    # python -m spacy download ru_core_news_lg && \
    # python -m spacy download es_dep_news_trf && \
    ###
    # Small spaCy tokenizers
    ###
    # python -m spacy download ca_core_news_sm && \
    # python -m spacy download zh_core_web_sm && \
    # python -m spacy download da_core_news_sm && \
    # python -m spacy download nl_core_news_sm && \
    # python -m spacy download en_core_web_sm && \
    # python -m spacy download fr_core_news_sm && \
    # python -m spacy download de_core_news_sm && \
    # python -m spacy download el_core_news_sm && \
    # python -m spacy download it_core_news_sm && \
    # python -m spacy download ja_core_news_sm && \
    # python -m spacy download lt_core_news_sm && \
    # python -m spacy download mk_core_news_sm && \
    # python -m spacy download xx_ent_wiki_sm && \
    # python -m spacy download nb_core_news_sm && \
    # python -m spacy download pl_core_news_sm && \
    # python -m spacy download pt_core_news_sm && \
    # python -m spacy download ro_core_news_sm && \
    # python -m spacy download ru_core_news_sm && \
    # python -m spacy download es_core_news_sm && \
    git clone https://github.com/huggingface/transformers.git /tmp/transformers

WORKDIR /tmp/transformers
RUN git reset --hard a4340d3b85fa8a902857d26d7870c53f82a4f666 && \ 
    pip install -e . && \
    pip install -r ./examples/pytorch/text-classification/requirements.txt