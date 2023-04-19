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