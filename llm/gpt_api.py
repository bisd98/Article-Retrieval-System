from openai import OpenAI

class OpenAIClient:
    _instance = None
    _client = None
    
    def __new__(self, api_key=None):
        """
        Singleton pattern to ensure only one instance of the OpenAIClient is created.
        """
        if self._instance is None:
            self._instance = super(OpenAIClient, self).__new__(self)
            if api_key:
                self._instance._client = OpenAI(api_key=api_key)
        return self._instance
    
    def openai_gpt(self, documents, query):
        """
        Generates a response to a user's query using the OpenAI GPT-3.5 Turbo model.

        Args:
            documents (list): A list of documents to provide context for generating the response.
            query (str): The user's query.

        Returns:
            str: The generated response to the user's query.
        """
        messages = [
            {
                "role": "system",
                "content": f"""DOCUMENTS:
                {"".join(documents)}

                QUESTION:
                {query}

                INSTRUCTIONS:
                Answer users QUESTION using the texts of the DOCUMENTS above.
                Keep your answer ground in the informations of the DOCUMENT.
                If the DOCUMENT does not contain information to answer the QUESTION,
                inform the USER that you do not have information on this subject """,
            },
            {"role": "user", "content": query},
        ]

        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            n=1,
            stop=None,
            temperature=0.7,
        )

        message = response.choices[0].message.content

        return message
