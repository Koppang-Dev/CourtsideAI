
# This is responsible for web scrapping and getting the images for the selected NBA players

# Importing necessary libraries
import os
import sys
import time
from simple_image_download import simple_image_download as simp

# Define the source and destination folders
SOURCE_FOLDER = "/home/riley.koppang/final_project/player_labelling/images/nba_images"

DESTINATION_FOLDER = "/home/riley.koppang/final_project/player_labelling/images/nba_player_images"

# Function to get a list of NBA players (example using a static list)
def get_nba_players():
    players = [
        # "Luka Doncic",
        # "Dorian Finney-Smith",
        # "Jordan Goodwin",
        # "Rui Hachimura",
        # "Jaxson Hayes",
        # "Bronny James",
        # "LeBron James",
        # "Trey Jemison III",
        # "Maxi Kleber",
        # "Dalton Knecht",
        # "Christian Koloko",
        # "Alex Len",
        "Shake Milton",
        "Markieff Morris",
        "Austin Reaves",
        "Cam Reddish",
        "Jarred Vanderbilt",
        "Gabe Vincent"
    ]
    return players

# Function to download images for each player
def download_images(player_names, limit=100):
    # Initialize the downloader
    response = simp.simple_image_download()


    # Downloading images for each player
    for player in player_names:
        print(f"Downloading images for: {player}")
        
        # Download images for the player
        response.download(player, limit=limit)

        # Move images to a folder named after the player in the source folder
        player_folder = os.path.join(SOURCE_FOLDER, player.replace(" ", "_"))
        os.makedirs(player_folder, exist_ok=True)

        # Move downloaded images to the player's folder in the source folder
        downloaded_folder = os.path.join("simple_images", player.replace(" ", "_"))
        if os.path.exists(downloaded_folder):
            for file_name in os.listdir(downloaded_folder):
                file_path = os.path.join(downloaded_folder, file_name)
                if os.path.isfile(file_path):
                    os.rename(file_path, os.path.join(player_folder, file_name))
                    print(f"Moved {file_name} to {player_folder}")
            # Remove the empty folder
            os.rmdir(downloaded_folder)
            print(f"Removed empty folder: {downloaded_folder}")

# Main function
def main():
    # Step 1: Get a list of NBA players
    player_names = get_nba_players()
    if not player_names:
        print("No players found. Exiting.")
        return

    # Step 2: Download images for each player
    download_images(player_names, limit=50)  



if __name__ == "__main__":
    main()