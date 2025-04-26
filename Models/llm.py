import instructor
from google.genai import types
from google import genai
from instructor import Mode


from config.settings import GOOGLE_API_KEY, GEMINI_MODEL_NAME



generate_content_config = types.GenerateContentConfig(
        temperature=0.15,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
    )


inference_config = {"temperature": 0.15}


if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    instructor_client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS,config=generate_content_config)
else:
    print("Error: Google API Key not configured. Gemini functionality will be disabled.")
    instructor_client = None 




def generate_response(system_prompt, response_model):
    if instructor_client:
        response = instructor_client.chat.completions.create(
            model=GEMINI_MODEL_NAME, 
            messages=[
                {"role": "user", "content": system_prompt},
            ],
            response_model=response_model,
            max_retries=3
        )
        return response
    else:
        raise ValueError("Gemini client not configured due to missing API key.")

# Example usage
# response = generate_response(system_prompt)