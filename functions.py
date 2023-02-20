import re
def extract_contact_info(sentence):
    # Address pattern
    address_pattern = r"[0-9]{1,3} .+, .+, [A-Z]{2} [0-9]{5}"
    address_match = re.search(address_pattern, sentence)
    address = address_match.group(0) if address_match else None

    # Phone number pattern
    phone_pattern =  r"\+?(\d{2})?\s*[-]?\s*(\d{10})"
    phone_match = re.search(phone_pattern, sentence)
    phone = phone_match.group(0) if phone_match else None

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_match = re.search(email_pattern, sentence)
    email = email_match.group(0) if email_match else None

    # Output
    output = f"Address is: {address}\n" if address else ""
    output += f"Mobile is: {phone}\n" if phone else ""
    output += f"Email is: {email}" if email else ""

    return output
def extract_airport_codes(sentence):
    airport_codes = {
        "JFK": "New York",
        "LAX": "Los Angeles",
        "ORD": "Chicago",
        "DFW": "Dallas",
        "DEN": "Denver"
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
    return source_code,destination_code