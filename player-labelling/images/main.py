
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

        # Clippers Players
        "Kawhi leonard",
        "Ivica Zubac",
        "Kris Dunn",
        "Norman Powell",
        "James Harden",

        # Lakers Players
        "Luka Doncic",
        "Dorian Finney-Smith",
        "Jaxson Hayes",
        "LeBron James",
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
        search_query = player + " NBA player"

        response.download(search_query, limit=limit)

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