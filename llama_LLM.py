import requests

def trimite_catre_llama(prompt_text):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:latest",  
        "prompt": (
            "You are a NAO robot that interprets commands and responds with an action.\n"
            "You will receive a message or command and will need to respond with one of the permitted codes.\n"
            "Permitted codes: [GREET, RAISE_RIGHT_ARM, RAISE_LEFT_ARM, WAVE, SIT, STAND, STOP].\n"
            f"Command: \"{prompt_text}\"\n"
            "Response (exact action code):"
        ),
        "stream": False
    }

    response = requests.post(url, json=payload)
    
    data = response.json()
    
    return data["response"].strip()
