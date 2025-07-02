import json

def load_celpip_data():
    try:
        with open('celpip.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("Error: celpip.json file not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in celpip.json")
        return None

def main():
    # Load the data
    words_data = load_celpip_data()
    
    if words_data:
        # Print the number of words loaded
        print(f"Successfully loaded {len(words_data)} words")
        
        # Example: Print first word's details
        if words_data:
            first_word = words_data[0]
            print("\nExample word details:")
            print(f"Word: {first_word['word']}")
            print(f"Ukrainian: {first_word['ua']}")
            print(f"Meaning: {first_word['meaning']}")
            print(f"Number of examples: {len(first_word['examples'])}")
