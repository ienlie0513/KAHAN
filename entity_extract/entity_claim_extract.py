import logging
import argparse
import numpy as np
import pandas as pd

import pywikibot
from tqdm import tqdm

from tqdm.contrib.concurrent import process_map
import ray

# get news content and comments from preprocessed tcv file
def get_data(df):
    f_en = []
    r_en = []
    for idx in range(df.id.shape[0]):
        for ens in df.entities[idx].split('||'):
            for en in ens.split(' '):
                if (en != ''):
                    if df.label[idx] == 0:
                        f_en.append(en)
                    if df.label[idx] == 1:
                        r_en.append(en)

    print ("fake unique entities: ", len(set(f_en)))
    print ("real unique entities: ", len(set(r_en)))

    unique_entities = list(set(f_en + r_en))

    return unique_entities

# given entity, return item dict, else None
def get_entity(site, entity):
    page = pywikibot.Page(site, entity)
    try: 
        item = pywikibot.ItemPage.fromPage(page)
    except pywikibot.exceptions.NoPageError:
        logging.error("NoPageError: entity {} not found".format(entity))
        return None
    except pywikibot.exceptions.MaxlagTimeoutError:
        logging.error("MaxlagTimeoutError: entity {} time out".format(entity))
        return None

    return item.get()

# given claim dict of an entity, return labels of claims
def get_claims(en, clm_dict):
    claims = []
    for clm_id, clms in clm_dict.items(): 
        for clm in clms:
            try: 
                clm_trgt = clm.getTarget()
            except pywikibot.exceptions.NoPageError:
                clm_trgt = None
                logging.error("NoPageError: entity {} claim {}".format(en, clm_id))
            # if not string identifiers, and images
            if isinstance(clm_trgt, pywikibot.page.ItemPage):
                try:
                    clm_dict = clm_trgt.get()
                    if "en" in clm_dict["labels"]:
                        claims.append(clm_dict["labels"]["en"])
                except pywikibot.exceptions.NoPageError:
                    logging.error("NoPageError: entity {}:{} claim {}".format(en, clm.on_item, clm_id))
                except pywikibot.exceptions.IsRedirectPageError:
                    logging.error("IsRedirectPageError: entity {}:{} claim {}".format(en, clm.on_item, clm_id))
                except pywikibot.exceptions.APIMWError:
                    logging.error("APIMWError: entity {}:{} claim {}".format(en, clm.on_item, clm_id))
                except pywikibot.exceptions.MaxlagTimeoutError:
                    logging.error("MaxlagTimeoutError: entity {}:{} claim {}".format(en, clm.on_item, clm_id))
    return '||'.join(claims)

@ray.remote
def get_entity_claims_batch(entities):
    site = pywikibot.Site("en", "wikipedia")
    clm_list = []
    for en in entities:
        item_dict = get_entity(site, en)
        clms = get_claims(en, item_dict["claims"]) if item_dict else ""
        clm_list.append({"entity": en, "claims": clms})
    return clm_list
# def get_entity_claim(en):
#     site = pywikibot.Site("en", "wikipedia")
#     item_dict = get_entity(site, en)
#     clms = get_claims(en, item_dict["claims"]) if item_dict else ""
#     return {"entity": en, "claims": clms}

def entity_claim_extract(entities, batch_size_per_worker):

    # Split entities into equally sized batches for each worker
    entity_batches = [entities[i:i + batch_size_per_worker] for i in range(0, len(entities), batch_size_per_worker)]

    object_refs = [get_entity_claims_batch.remote(batch) for batch in entity_batches]
    clm_list = []

    with tqdm(total=len(entities)) as progress_bar:
        while object_refs:
            ready_object_refs, object_refs = ray.wait(object_refs, num_returns=1)
            results = ray.get(ready_object_refs[0])
            clm_list.extend(results)
            progress_bar.update(len(results))

    return pd.DataFrame(clm_list)
# def entity_claim_extract(entities):
#     site = pywikibot.Site("en", "wikipedia")

#     clm_list = []
#     for en in tqdm(entities):
#         item_dict = get_entity(site, en)
#         clms = get_claims(en, item_dict["claims"]) if item_dict else ""
#         clm_list.append({"entity": en, "claims": clms})

#     return pd.DataFrame(clm_list)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="politifact")
    parser.add_argument("--batch_size_per_worker", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(filename='claim_{}.log'.format(args.platform), level=logging.INFO)
    logging.info("Entity claims extraction start")
    
    logging.info("Get entity data")
    df = pd.read_csv("./data/{}_no_ignore_en.tsv".format(args.platform), sep='\t') 
    df = df.fillna('')
    entities = get_data(df)

    logging.info("Extract entity claims")
    # clm_df = entity_claim_extract(entities)
    clm_df = entity_claim_extract(entities, args.batch_size_per_worker)

    logging.info("Output extracted claims to tsv file")
    clm_df.to_csv("./data/{}_no_ignore_clm.tsv".format(args.platform), sep = '\t', index=False)

