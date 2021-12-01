import json
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
import argparse
import json

def load_data():
    hetjson=json.load(open("hetionet-v1.0.json"))
    nodes=hetjson['nodes']
    return nodes

def get_anatomy_description(node):
    """
    Get text about 'Anatomy' entities
    """
    if 'url' not in node['data'].keys():
        description = ""
    else:
        url = node['data']['url']
        page = requests.get(url)
        soup = BeautifulSoup(page.content,"html.parser")
        try: 
            description = soup.find("div", property="description").span.text
        except AttributeError:
            description = ""
    k = node['kind']+"::"+node['identifier']
    D[k] = description
    return description

def get_bp_cc_mf_description(node):
    """
    Get text about 'Biological Process', 'Cellular Component', 'Disease', 'Molecular Function' entities
    """
    if 'url' not in node['data'].keys():
        description = ""
    else:
        url = node['data']['url']
        page = requests.get(url)
        soup = BeautifulSoup(page.content,"html.parser")
        if soup.find("ns3:iao_0000115") is not None:
            description = soup.find("ns3:iao_0000115").text
        elif soup.find("ns4:iao_0000115") is not None:
            description = soup.find("ns4:iao_0000115").text
        else:
            description = ""
    k = node['kind']+"::"+node['identifier']
    D[k] = description
    return description

def get_cp_description(node):
    if 'url' not in node['data'].keys():
        description = ""
    else:
        url = node['data']['url']
        page = requests.get(url)
        soup = BeautifulSoup(page.content,"html.parser")
        if soup.find("dt", {"id": "summary"}):
            description = soup.find("dt", {"id": "summary"}).nextSibling.text
        else:
            description = ""
    k = node['kind']+"::"+node['identifier']
    D[k] = description
    return description
    
def get_pw_description(node):
    """
    Get text about 'Pathway'
    """
    if 'url' not in node['data'].keys():
        description = ""
    else:
        url = node['data']['url']
        page = requests.get(url)
        soup = BeautifulSoup(page.content,"html.parser")
        if soup.find("div", {"id":"descr"}):
            description = soup.find("div", {"id":"descr"}).text
        else:
            description = ""
    k = node['kind']+"::"+node['identifier']
    D[k] = description
    return description

def get_pc_description(node):
    if 'url' not in node['data'].keys():
        description = ""
    else:
        url = node['data']['url']
        page = requests.get(url)
        soup = BeautifulSoup(page.content,"html.parser")
        if soup.table:
            description = ""
            for r in soup.table.find_all("tr"):
                for d in r.find_all("td"):
                    if d.text.lower() == 'mesh definition':
                        description = d.parent.p.text
                        break
        else:
            description = ""
    k = node['kind']+"::"+node['identifier']
    D[k] = description
    return description

def get_text(nodes, func):
    start = time.time()
    threads = 13
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(func, nodes, chunksize=max(1, len(nodes)//500))
    print(time.time()-start)

def write_text(fname, dic):
    with open("{}.json".format(fname), "w") as f:
        jsonfile = json.dumps(dic, indent=4)
        f.write(jsonfile)

if __name__ == "__main__":
#  'Anatomy', 'Biological Process', 'Cellular Component', 'Compound', 'Disease',
#  'Gene'(local), 'Molecular Function', 'Pathway', 'Pharmacologic Class',
#  'Side Effect'(NA),'Symptom'(NA)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", help="Hetionet Entity Type Name")
    args = parser.parse_args()

    all_nodes = load_data()
    nodes_dict = {}
    for n in all_nodes:
        nodes_dict.setdefault(n['kind'].lower(),[]).append(n)
    
    fname = args.name.lower()
    nodes = nodes_dict[fname]
    if fname == "anatomy":
        D = {}
        get_text(nodes, get_anatomy_description)
        write_text(fname, D)
    elif fname in ["biological process", "cellular component", "disease", "molecular function"]:
        D = {}
        get_text(nodes, get_bp_cc_mf_description)
        write_text(fname, D)
    elif fname == "pathway":
        D = {}
        get_text(nodes, get_pw_description)
        write_text(fname, D)
    elif fname == "pharmacologic class":
        D = {}
        get_text(nodes, get_pc_description)
        write_text(fname, D)
    elif fname == 'gene':
        D = {}
        _ = [D.setdefault(n['kind']+"::"+str(n['identifier']), n['data']['description']) for n in nodes]
        write_text(fname, D)
    elif fname == 'compound':
        D = {}
        get_text(nodes, get_cp_description)
        write_text(fname, D)
