# import packages
import sys
import os
from google_images_search import GoogleImagesSearch

# configure scraper wit API key, engine_id
gis = GoogleImagesSearch('AIzaSyCgp1e7FuuDrFXydzYqvhYGUdquyi7oIiw', 'f478206a362214c27')

"""
google_image_scrape: 

--> Method to scrape domain restricted images for a particular search_term

--> Inputs: 
  - search_term, a string defining the query (e.g. 'Ukraine') 
  - website_domain, a string defining the website domain (e.g. 'nytimes.com') 
  - num_of_images, an integer defining the number of images to be scraped (e.g. 10)
  - output_path, a string defining a path to save the results to (e.g. '../data') 
  
--> Outputs: 
  -  
"""
def google_image_scrape(search_term, website_domain, num_of_images, output_path):

    # build path to store images
    path = '{}/{}.{}/'.format(output_path, website_domain, search_term.replace(" ", "_"))

    if not os.path.exists(path):
        os.mkdir('{}/{}.{}'.format(output_path, website_domain, search_term.replace(" ", "_")))
        os.chmod('{}/{}.{}'.format(output_path, website_domain, search_term.replace(" ", "_")), 0o777)

    # loop to download photos
    count = 0

    _search_params = {
        'q': search_term + ' ' + 'site:' + website_domain,
        'num': num_of_images,
    }

    # send request
    gis.search(search_params=_search_params, path_to_dir=path)

    # iterate through the results
    while count < num_of_images:
        gis.next_page()
        count += 10