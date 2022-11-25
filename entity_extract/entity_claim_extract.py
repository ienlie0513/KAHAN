import logging
import argparse
import numpy as np
import pandas as pd

import pywikibot
from tqdm import tqdm


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

def entity_claim_extract(entities):
    site = pywikibot.Site("en", "wikipedia")

    clm_list = []
    for en in tqdm(entities):
        item_dict = get_entity(site, en)
        clms = get_claims(en, item_dict["claims"]) if item_dict else ""
        clm_list.append({"entity": en, "claims": clms})

    return pd.DataFrame(clm_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="politifact")
    args = parser.parse_args()

    logging.basicConfig(filename='claim_{}.log'.format(args.platform), level=logging.INFO)
    logging.info("Entity claims extraction start")
    
    logging.info("Get entity data")
    df = pd.read_csv("./data/{}_no_ignore_en.tsv".format(args.platform), sep='\t') 
    df = df.fillna('')
    entities = get_data(df)

    logging.info("Extract entity claims")
    clm_df = entity_claim_extract(entities)

    logging.info("Output extracted claims to tsv file")
    clm_df.to_csv("./data/{}_no_ignore_clm.tsv".format(args.platform), sep = '\t', index=False)

