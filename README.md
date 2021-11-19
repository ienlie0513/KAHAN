This repository is the implementation of KAHAN: Knowledge-Aware Hierarchical Attention Network for Fake News detection on Social Media. 

---
#### Code
mian.py - Main function for executing code. Involves loading dataset and pre processing data including train test split.

KAHAN.py - Involves model construction, training and evaluate. 

---
#### Entity Extraction
1. Entity extraction: We use [REL](https://github.com/informagi/REL) as our entity linking model to extract entities from news content. The wikidata version is 2019 and the NER model is Flair. The related code is at entity_extract.py.
2. Entity claim extraction: We use [pywikibot](https://github.com/wikimedia/pywikibot) to query entity claims in Wikidata. The related code is at entity_claim_extract.py.

---
#### Dataset
The experimentation and results are for the FakeNewsNet dataset. Due to privacy policies in Twitter, [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) is not publicly disclosed yet.
The dataset can be obtained upon request for research and non-commercial purposes.

To facilitate the reproduction of experimental results, we provide the preprocessed data.
* en file: the preprocessed data with entities extracted from news content
* clm file: the entity claims of the entities



