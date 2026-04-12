#%%
import requests
import os
import re
from urllib.parse import urljoin

def download_request(url,output_file):
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Download completed:", output_file)
        
def download_veg_from_mapbiomas():
    for year in range(1985,2025):
        
        url = f"https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{year}.tif"
        output_file = f"../data/veg/brazil_coverage_{year}.tif"

        download_request(url,output_file)
        
        
def download_nasa_susceptibility_map():
    url = "https://gpm.nasa.gov/sites/default/files/downloads/global-landslide-susceptibility-map-2-27-23.tif"
    output_file = "../data/deslizamento/nasa_susceptibility_map.tif"
    download_request(url,output_file)
    

    
# "VN", "HN"
def download_zip_from_topodata(suffixes=["ZN", "SN", "ON",]):
    # Base URL of the directory
    base_url = "https://www.dsr.inpe.br/topodata/data/geotiff/"

    # Local folder to save downloaded zips
    output_folder = "../data/topodata_zips"
    os.makedirs(output_folder, exist_ok=True)

    #Get directory listing
    print("Fetching directory listing...")
    resp = requests.get(base_url)
    resp.raise_for_status()
    # Extract all .zip links
    zip_files = re.findall(r'href="([^"]+\.zip)"', resp.text)

    # -----------------------------
    # Download matching files
    # -----------------------------
    for suffix in suffixes:
        pattern = rf"{suffix}\.zip$"
        matching_files = [f for f in zip_files if re.search(pattern, f)]
        print(f"Found {len(matching_files)} files matching pattern '{suffix}.zip'")
    
        cur_folder = os.path.join(output_folder, suffix)
        os.makedirs(cur_folder, exist_ok=True)
        
        for filename in matching_files:

            file_url = urljoin(base_url, filename)
            output_path = os.path.join(cur_folder, filename)

            if os.path.exists(output_path):
                print(f"Skipped (exists): {filename}")
                continue

            try:
                print(f"Downloading {filename} ...")
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f"Saved to: {output_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                with open("error_log.txt", "a") as f:
                        f.write(f"Error downloading {filename} from {file_url}: {str(e)}\n")

                
        print("All done!")

                