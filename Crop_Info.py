import requests
from bs4 import BeautifulSoup
import re
from PIL import Image

crop_info = {}

crops = {
    'Rice': ['June', 'July', 'August', 'September'],
    'Maize': ['June', 'July', 'August', 'September'],
    'Cotton': ['February', 'March', 'April'],
    'Coconut': ['May', 'June', 'July', 'August'],
    'Banana': ['February', 'March', 'August', 'September', 'October', 'November'],
    'Black Gram': ['August', 'September', 'October'],
    'Green Gram': ['August', 'September', 'October'],
    'Red Gram': ['August', 'September', 'October'],
    'Bengal Gram': ['October', 'November', 'December', 'January'],
    'Coffee': ['October', 'November', 'December', 'January', 'February']
}

for crop in crops:
    crop_dict = {}
    crop_dict['Crop'] = crop
    seasons = crops[crop]
    crop_dict['Seasons'] = seasons
    img_url = f'Downloads/{crop}.jpg'
    crop_dict['Image'] = img_url
    crop_info[crop] = crop_dict

for crop_in in crop_info :
    print ("\n " , crop_in , " : " , crop_info [ crop_in ] )
    img = Image.open (f'{crop_in}.jpeg')
    img.show()