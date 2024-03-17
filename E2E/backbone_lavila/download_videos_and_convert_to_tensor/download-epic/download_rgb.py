import requests
import os 
import re 
import shutil

from bs4 import BeautifulSoup



EPIC_KITCHEN_TORRENT_URL = 'https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m'


def extract_rgb_frame_links(url):
    """
    extracts from data bris's epic-kitchen torrents all the links 
    to the rgb frames of each participant 
    """
    links = []
    html_content = requests.get(url).content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find all <a> tags
    for link in soup.find_all('a'):
        if 'rgb_frames' in link.get_text().lower():  # Check if anchor text contains 'rgb_frames'
            href_value = link.get('href')
            links.append(href_value)
    print(links)

    return links


def extract_links_for_each_participant(url):
    links = []
    html_content = requests.get(url).content
    soup = BeautifulSoup(html_content , 'html.parser')
    for idx, link in enumerate(soup.find_all('a')):
        title_text = link.get_text().lower()
        print(title_text, idx)
        # Check if the title matches the pattern 'P[any_number]_[any_number].tar'
        if re.search(r'p\d+_\d+\.tar', title_text):
            links.append(link.get('href'))
    print(links)
    return links



def download_file(url, file_name):

    with requests.get(url, stream=True) as r:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def main(EPIC_KITCHEN_TORRENT_URL):
    
    rgb_frame_links = extract_rgb_frame_links(EPIC_KITCHEN_TORRENT_URL)
    for idx, participant_rgb_link  in enumerate(rgb_frame_links):
        if not os.path.exists(f'./images/P{idx}'):
            os.makedirs(f'./images/P{idx}')
        
        participant_links = extract_links_for_each_participant(f'https://data.bris.ac.uk/data/dataset/{participant_rgb_link}')
        for idx_2, link in enumerate(participant_links):
            download_file(link, f'./images/P{idx}/{idx_2}.tar')


if __name__ == '__main__':
    main(EPIC_KITCHEN_TORRENT_URL)
