from openai import OpenAI

def openai_gpt(api_key, documents, query):
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": f"""DOCUMENTS:
            {"".join(documents)}

            QUESTION:
            {query}

            INSTRUCTIONS:
            Answer users QUESTION using the texts of the DOCUMENTS above.
            Keep your answer ground in the informations of the DOCUMENT.
            If the DOCUMENT does not contain information to answer the QUESTION,
            inform the USER that you do not have information on this subject """},
        {"role": "user", "content": query},
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        n=1,
        stop=None,
        temperature=0.7,
    )

    message = response.choices[0].message.content
    
    return message