import re
import random
import translate
from langdetect import detect
import ast
import re

def extract_contact_info(text):
    # Regular expression pattern to match email addresses in the input text
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    
    # Regular expression pattern to match Indian mobile numbers in the input text
    mobile_pattern = r"\b[2-9]\d{9}\b"
    
    # Regular expression pattern to match Indian addresses in the input text
    address_pattern1 = r"\b\d{1,3}[-,/\s]*[A-Za-z0-9\s,-/]*[,]*\s*[A-Za-z0-9\s,-/]*[,]*\s*[A-Za-z0-9\s,-/]*[,]*\s*[A-Za-z0-9\s,-/]*\s*\b\d{6}\b"
    
    # Regular expression pattern to match Indian addresses in the input text (alternative format)
    address_pattern2 = r"\b(\d+\s[A-Za-z\s]+,\s[A-Za-z]+\s[A-Za-z]+\s\d{5})\b"

   # Regular expression pattern to match US addresses in the input text
    address_pattern3 = r"\b(\d+\s[A-Za-z\s]+\n[A-Za-z\s]*,\s[A-Za-z]+\s\d{5}(-\d{4})?)\b"
    
    # Regular expression pattern to match US addresses in the input text (alternative format)
    address_pattern4 = r"\b([A-Za-z]+\s\d{1,5},\s[A-Za-z]+,\s[A-Za-z]{2}\s\d{5})\b"
    
    # Extract email address from the input text
    email = re.findall(email_pattern, text)
    email = email[0] if email else None
    
    # Extract mobile number from the input text
    mobile = re.findall(mobile_pattern, text)
    mobile = mobile[0] if mobile else None
    
    # Extract address from the input text
    address = None
    city = None
    country = None
    zipcode = None
    # Try to extract the address using the first pattern
    match = re.search(address_pattern1, text, re.IGNORECASE)
    if match:
        address = match.group(0)
        city = match.group(4) if len(match.groups()) >= 5 else None
        country = match.group(5) if len(match.groups()) >= 6 else None
        #zipcode = match.group(5)
    else:
        # Try to extract the address using the second pattern
        match = re.search(address_pattern2, text, re.IGNORECASE)
        if match:
            address = match.group(0)
            city = None
        else:
            # Try to extract the address using the third pattern
            match = re.search(address_pattern3, text, re.IGNORECASE)
            if match:
                address = match.group(0)
                city = match.group(2).split(",")[0] if len(match.groups()) >= 3 else None
                country = "US"
    
    # Return a dictionary containing the extracted contact information
    return {"email": email, "mobile": mobile, "address": address, "city": city, "country": country}

def translate_text(text):
    text=text.lower()
    keywords_ls=['translate','convert']
    for keyword in keywords_ls:
        if keyword in text:
            index=text.find(keyword)
            #text_trans=text[index+len(i):].strip()
            break
    else:
            return "Please provide valid input"
    target_lang=None
    for lang in ['french','german','hindi','punjabi','dutch','bengali','chinese','latin','portuguese']:
        for preposition in ['into','to','in']:
            pattern=fr"{keyword}\s+(.*?)\s+{preposition}\s+{lang}"
            match=re.search(pattern,text)
            if match:
                text_trans=match.group(1)
                target_lang=lang
                break
        if target_lang is not None:
            break
    if target_lang is None:
            text_trans=text[index+len(keyword):].strip()
            target_lang=detect(text_trans)
    
    translator=translate.Translator(to_lang=target_lang)
    transl=translator.translate(text_trans)
    return transl
def translat_text(text):
    target_language='en'
    text=text.lower()
    if 'into french' in text:
        target_language='fr'
    elif 'into german' in text:
        target_language='de'
    elif 'into hindi' in text:
        target_language='hi'
    elif 'into german' in text:
        target_language='de'
    elif 'into punjabi' in text:
        target_language='pa'
    elif 'into chinese' in text:
        target_language='zh-cn'
    elif 'into urdu' in text:
        target_language='ur'
    else:
        target_language=detect(text)
    key_ls=['translate','convert']
    for i in key_ls:
        if i in text:
            index=text.find(i)
            text_to=text[index + len(i):].strip()
            break
    else :
        return "Invalid Input"

    translator=translate.Translator(to_lang=target_language)
    translation=translator.translate(text_to)
    return translation

################-----------------First to third person------------------------------------##############
def first_to_third_person(sentence):
    # split sentence into words
    words = sentence.split()
    
    # list of pronouns to replace
    first_person_pronouns = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    
    # dictionary of replacements
    replacements = {
        'I': 'he' if random.randint(0, 1) == 0 else 'she',
        'me': 'him' if random.randint(0, 1) == 0 else 'her',
        'my': 'his' if random.randint(0, 1) == 0 else 'her',
        'mine': 'his' if random.randint(0, 1) == 0 else 'hers',
        'we': 'they',
        'us': 'them',
        'our': 'their',
        'ours': 'theirs'
    }
    
    # iterate through words, replacing first person pronouns with third person
    new_words = []
    for word in words:
        if word in first_person_pronouns:
            new_words.append(replacements[word])
        else:
            new_words.append(word)
            
    # join words back into a sentence
    new_sentence = ' '.join(new_words)
    
    return new_sentence

################----------------Airport Code Extraction----------------------------------##############
def extract_airport_codes(sentence):
    airport_codes = {
        "JFK": "New York",
        "LAX": "Los Angeles",
        "ORD": "Chicago",
        "DFW": "Dallas",
        "DEN": "Denver",'IXA':'Agartala',
'AGR':'Agra','AMD':'Ahmedabad','AJL':'Aizawl','ATQ':'Amritsar','IXU':'Aurangabad','IXB':'Bagdogra','BEK':'Bareilly','IXG':'Belagavi','BLR':'Bengaluru',
'BHO':'Bhopal','BBI':'Bhubaneswar','IXC':'Chandigarh','MAA':'Chennai','CJB':'Coimbatore','DBR':'Darbhanga','DED':'Dehradun','DEL':'Delhi','DGH':'Deoghar',
'DIB':'Dibrugarh','DMU':'Dimapur','RDP':'Durgapur','GAY':'Gaya','GOI':'Goa','GOP':'Gorakhpur','GAU':'Guwahati','GWL':'Gwalior',
'HBX':'Hubli','HYD':'Hyderabad','IMF':'Imphal','IDR':'Indore','HGI':'Itanagar','JLR':'Jabalpur','JAI':'Jaipur','IXJ':'Jammu','JDH':'Jodhpur','JRH':'Jorhat',
'CDP':'Kadapa','CNN':'Kannur','KNU':'Kanpur','COK':'Kochi','KLH':'Kolhapur','CCU':'Kolkata','CCJ':'Kozhikode','KJB':'Kurnool','IXL':'Leh','LKO':'Lucknow','IXM':'Madurai',
'IXE':'Mangaluru','BOM':'Mumbai','MYQ':'Mysuru','NAG':'Nagpur','GOX':'North Goa','PGH':'Pantnagar','PAT':'Patna','IXZ':'Port-Blair','IXD':'Prayagraj','PNQ':'Pune',
'RPR':'Raipur','RJA':'Rajahmundry','IXR':'Ranchi','SHL':'Shillong','SAG':'Shirdi','IXS':'Silchar','SXR':'Srinagar','STV':'Surat','TRV':'Thiruvrappalli','TIR':'Tirupati',
'TCR':'Tuticorin','UDR':'Udaipur','BDQ':'Vadodara',
'TRZ':'Tiruchirappalli','VNS':'Varanasi','VGA':'Vijayawada','VTZ':'Visakhapatnam','CNN':'Kannur','IXA':'Agartala',
'AMD':'Ahmedabad','ATQ':'Amritsar','IXB':'Bagdogra',
'BLR':'Bengaluru','BBI':'Bhubaneswar','BHO':'Bhopal','IXC':'Chandigarh','MAA':'Chennai','CJB':'Coimbatore','DED':'Dehradun','DEL':'Delhi','DMU':'Dimapur',
'GOI':'Goa','GOP':'Gorakhpur','GAU':'Guwahati','HBX':'Hubli','HYD':'Hyderabad','IMF':'Imphal','IDR':'Indore','JLR':'Jabalpur',
'JAI':'Jaipur','IXJ':'Jammu','CCU':'Kolkata','CCJ':'Kozhikode','LKO':'Lucknow','IXM':'Madurai','IXE':'Mangaluru',
'BOM':'Mumbai','NAG':'Nagpur','PAT':'Patna','PNQ':'Pune','RPR':'Raipur','RJA':'Rajahmundry','IXR':'Ranchi',
'SXR':'Srinagar','STV':'Surat','TRV':'Thiruvananthapuram','TRZ':'Tiruchirappalli','TCR':'Tuticorin','UDR':'Udaipur',
'BDQ':'Vadodara','VNS':'Varanasi','VGA':'Vijayawada','VTZ':'Visakhapatnam','GAY':'Gaya','JDH':'Jodhpur','IXS':'Silchar','IXG':'Belagavi','YVR': 'Vancouver',
'YYC': 'Calgary',
    'YEG': 'Edmonton',
    'YWG': 'Winnipeg',
    'YYZ': 'Toronto',
    'YOW': 'Ottawa',
    'YUL': 'Montreal',
    'YQB': 'Quebec City',
    'YHZ': 'Halifax',
    'YYT': 'St. John',
	"ATL": "Atlanta",
    "BOS": "Boston",
    "CLT": "Charlotte",
    "ORD": "Chicago",
    "CVG": "Cincinnati",
    "DFW": "Dallas/Fort Worth",
    "DEN": "Denver",
    "DTW": "Detroit",
    "HNL": "Honolulu",
    "IAH": "Houston",
    "LAS": "Las Vegas",
    "LAX": "Los Angeles",
    "MEM": "Memphis",
    "MIA": "Miami",
    "MSP": "Minneapolis",
    "JFK": "New York City",
    "EWR": "Newark",
    "LGA": "New York City",
    "OAK": "Oakland",
    "MCO": "Orlando",
    "PHL": "Philadelphia",
    "PHX": "Phoenix",
    "PDX": "Portland",
    "SLC": "Salt Lake City",
    "SAN": "San Diego",
    "SFO": "San Francisco",
    "SJC": "San Jose",
    "SEA": "Seattle",
    "STL": "St. Louis",
    "TPA": "Tampa",
    "DCA": "Washington, D.C.",
	"LHR": "London Heathrow Airport",
    "LGW": "London Gatwick Airport",
    "STN": "London Stansted Airport",
    "LTN": "London Luton Airport",
    "MAN": "Manchester Airport",
    "BHX": "Birmingham Airport",
    "GLA": "Glasgow Airport",
    "EDI": "Edinburgh Airport",
    "NCL": "Newcastle Airport",
    "BRS": "Bristol Airport",
    "LPL": "Liverpool John Lennon Airport",
    "EMA": "East Midlands Airport","ABZ": "Aberdeen Airport","BOH": "Bournemouth Airport",
    "DSA": "Doncaster Sheffield Airport","EXT": "Exeter Airport","JER": "Jersey Airport",
    "LBA": "Leeds Bradford Airport","LCY": "London City Airport","MME": "Durham Tees Valley Airport","NWI": "Norwich Airport","SOU": "Southampton Airport",
    "PIK": "Glasgow Prestwick Airport","SEN": "London Southend Airport","SYY": "Stornoway Airport","SWS": "Swansea Airport"
    }

    # Regular expression pattern to match city names in the input sentence
    city_pattern = r"\b(" + "|".join(airport_codes.values()) + r")\b"

    # Extract source and destination city names from the input sentence
    matches = re.findall(city_pattern, sentence)
    source_city = matches[0]
    destination_city = matches[1]

    # Get the corresponding airport codes for the source and destination cities
    source_code = [code for code, city in airport_codes.items() if city == source_city][0]
    destination_code = [code for code, city in airport_codes.items() if city == destination_city][0]
    return f"Source city is {source_code}"+ " \n " + f"Destination city is {destination_code}"