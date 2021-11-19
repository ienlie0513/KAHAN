This repository is the implementation of KAHAN: Knowledge-Aware Hierarchical Attention Network for Fake News detection on Social Media. 

---
#### Code
mian.py - Main function for executing code. Involves loading dataset and pre processing data including train test split.

KAHAN.py - Involves model construction, training and evaluate. 

config.json - The model and training setting, including the hyperparameters and the pre-trained word2vec and wikipedia2vec.

---
#### Dataset
The experimentation and results are for the FakeNewsNet dataset. Due to privacy policies in Twitter, [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) is not publicly disclosed yet.
The dataset can be obtained upon request for research and non-commercial purposes.

Below are the steps we preprocessed the data:
1. We aggregate news content, label and user comments related to the piece of news, and build the time index of each post. In additional, we filter data for less than three comments. The related code is at util/data_util.py.
2. We extract the entities from the news content and use [REL](https://github.com/informagi/REL) as our entity linking model. The wikidata version is 2019 and the NER model is Flair. The related code is at entity_extract.py.
3. We extract the entity claims of each entity from knowledge graph. We use [pywikibot](https://github.com/wikimedia/pywikibot) to query entity claims in Wikidata. The related code is at entity_claim_extract.py.



