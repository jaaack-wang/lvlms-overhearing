import re, os
import base64
from io import BytesIO
from PIL import Image

import filetype
from groq import Groq
from time import sleep
from litellm import completion
from litellm import supports_vision


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_resized_image(image_path, max_dim=1024):
    with Image.open(image_path) as img:
        # Resize image so the longest side is `max_dim`, preserving aspect ratio
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        # Save to a buffer in memory
        buffer = BytesIO()
        img.save(buffer, format="PNG")  # Use "JPEG" if you prefer that format
        buffer.seek(0)

        # Encode image to base64
        return base64.b64encode(buffer.read()).decode("utf-8")



def get_groq_complemtion(model, messages, temperature):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


class LVLMChat:

    def __init__(self, 
                 model="openai/gpt-4o-mini", 
                 completion_fn=None,
                 system_prompt=None, 
                #  messages=None, 
                 max_img_dim=None, 
                 temperature=0, 
                 max_tries=5):

        if model.startswith("groq/"):
            model = model.replace("groq/", "")
            completion_fn = lambda messages, temperature: get_groq_complemtion(
                model=model,
                messages=messages,
                temperature=temperature,
            )

        if completion_fn is None:
            assert supports_vision(model), f"Model {model} does not support vision."
            self.completion_fn = lambda messages, temperature: completion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        else:
            self.completion_fn = completion_fn

        # if messages is not None:
        #     self.messages = messages
        #     system_prompt = None
        # else:
        #     self.messages = []

        self.model = model
        self.messages = []
        self.max_img_dim = max_img_dim
        self.temperature = temperature
        self.max_tries = max_tries

        if system_prompt is not None:
            self.messages.append(self.__construct_message("system", system_prompt))

    def __construct_message_segment(self, text=None, image_path=None):
        assert text is not None or image_path is not None, "Either text or image_path should be provided"

        if text is not None:
            return {"type": "text", "text": text}

        if image_path is not None:
            if self.max_img_dim is None:
                base64_image = encode_image(image_path)
            else:
                base64_image = encode_resized_image(image_path, 
                                                    self.max_img_dim)

            extension = filetype.guess_extension(image_path).lower()
            if extension not in ["jpg", "jpeg", "png"]:
                raise ValueError(f"Unsupported image format: {extension}")

            if extension == "jpg":
                extension = "jpeg"

            return {"type": "image_url",
                    "image_url": {"url": f"data:image/{extension};base64,{base64_image}"}}

    def __construct_message(self, role, prompt):
        '''Construct a message for the chat given the role and prompt.'''

        assert role in ["user", "assistant", "system"], "Role must be either 'user,' 'assistant,' or 'system.'"

        if role == "system":
            return {"role": role, "content": prompt}
        
        if "groq/" in self.model and role == "assistant":
            return {"role": role, "content": prompt}

        content = []
        segments = re.findall(r"[^<>]+|<[^>]+>", prompt)

        for segment in segments:
            if len(segment.strip()) == 0:
                continue
            elif "<" in segment and ">" in segment:
                image_path = segment[1:-1]

                if filetype.is_image(image_path):
                    content.append(self.__construct_message_segment(image_path=image_path))
                else:
                    raise ValueError(f"Invalid image path: {image_path}")                                
            else:
                content.append(self.__construct_message_segment(text=segment))

        if len(content) == 0:
            raise ValueError("Message content is empty after processing.")

        return {"role": role, "content": content}

    def __get_completion(self):
        assistant_response = None

        for _ in range(self.max_tries):

            try:
                assistant_response = self.completion_fn(
                    self.messages, self.temperature)
                break
            except Exception as e:
                print("Running into problem:", e)
                print("Retrying...")
                sleep(10)

        if assistant_response is None:
            assistant_response = f"SOMETHING WRONG"
        else:
            assistant_response = assistant_response.choices[0].message.content
            self.messages.append(self.__construct_message("assistant", assistant_response))

        return assistant_response

    def get_chat_completion(self, prompt):
        self.messages.append(self.__construct_message("user", prompt))
        return self.__get_completion()

    def get_chat_completion_from_messages(self, messages):
        self.messages = messages
        return self.__get_completion()