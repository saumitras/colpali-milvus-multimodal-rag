import requests
import os
import google.generativeai as genai

from typing import List
from utils import encode_image
from PIL import Image

class Rag:

    def get_answer_from_gemini(self, query, imagePaths):

        print(f"Querying Gemini for query={query}, imagePaths={imagePaths}")

        try:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            images = [Image.open(path) for path in imagePaths]
            
            chat = model.start_chat()

            response = chat.send_message([*images, query])

            answer = response.text

            print(answer)
            
            return answer
        
        except Exception as e:
            print(f"An error occurred while querying Gemini: {e}")
            return f"Error: {str(e)}"
        

    def get_answer_from_openai(self, query, imagesPaths):
        print(f"Querying OpenAI for query={query}, imagesPaths={imagesPaths}")

        try:    
            payload = self.__get_openai_api_payload(query, imagesPaths)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
    
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
    
            answer = response.json()["choices"][0]["message"]["content"]
    
            print(answer)
    
            return answer
    
        except Exception as e:
            print(f"An error occurred while querying OpenAI: {e}")
            return None


    def __get_openai_api_payload(self, query:str, imagesPaths:List[str]):
        image_payload = []

        for imagePath in imagesPaths:
            base64_image = encode_image(imagePath)
            image_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        *image_payload
                    ]
                }
            ],
            "max_tokens": 1024
        }

        return payload
    


# if __name__ == "__main__":
#     rag = Rag()
    
#     query = "Based on attached images, how many new cases were reported during second wave peak"
#     imagesPaths = ["covid_slides_page_8.png", "covid_slides_page_8.png"]
    
#     rag.get_answer_from_gemini(query, imagesPaths)