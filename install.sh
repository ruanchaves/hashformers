pip install lm_scorer==0.4.2
pip install git+git://github.com/ruanchaves/mlm-scoring.git@master#egg=mlm
pip install spacy==3.1.1
pip install emoji==1.4.2 python-json-logger==2.0.2

for value in en_core_web_sm fr_core_news_sm de_core_news_sm it_core_news_sm xx_ent_wiki_sm pt_core_news_sm es_core_news_sm
do
    python -m spacy download $value
done

rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cd ./transformers
git reset --hard a4340d3b85fa8a902857d26d7870c53f82a4f666
pip install -e .
pip install -r ./examples/pytorch/text-classification/requirements.txt
cd ..