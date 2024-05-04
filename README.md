# Article-Retrieval-System

**Description**

A system that indexes articles from the "1300 Towards Data Science Medium Articles" dataset available on Kaggle, applying Retrieval Augmented Generation (RAG) for the effective retrieval of article fragments in response to queries.

**Features**

- Retrieves predefined chunks of information based on user queries.
- Integrates with OpenAI's GPT-3.5 Turbo model to generate responses beyond basic retrieval.
- Chat-like interface for interacting with the system.
- Allows users to activate the RAG system by providing their OpenAI API key.
- Provides a button to gracefully shut down the system, closing the database connection and terminating the session.

**Requirements**

- Python 3.10 or higher

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bisd98/Article-Retrieval-System.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Article-Retrieval-System
   ```

3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the system, use the following command in the project directory:

```bash
streamlit run user_interface.py
```

After running the system using the command mentioned above, the following processes will be initiated:

1. Chunking and Vectorization:
- The system will perform chunking and vectorization of data, breaking down input into manageable "chunks" and converting them into vector representations.
2. Vector Database Initialization:
- Initialization of the vector database will take place, preparing it to store and retrieve vectorized data efficiently.
3. Data Loading into Vector Database:
- Data will be loaded into the vector database, allowing for quick access during user interactions.

### System setup

During the system setup, the user interface will display information about data loading. Once these processes are completed, interaction with the system will become available. The progress of each stage in setting up the system will be shown in the terminal output.

![Example Image](https://i.ibb.co/bmp78wT/setup-system.png)

Ensure that you monitor the terminal output for the sequence and completion of these setup steps before interacting with the system through the user interface.

### Retrieval system

The interface resembles a chat with the system. The primary functionality involves retrieving chunks via the retrieval system.

![Example Image](https://i.ibb.co/TLWpSJj/only-retrieval.png)

### Retrieval Augmented Generation

Additionally, users can leverage the RAG system based on GPT-3.5 Turbo. To activate RAG, enter your OpenAI API key in the sidebar panel.

![Example Image](https://i.ibb.co/s6TSv1r/rag.png)

### Close system

To deactivate system, click the 'Shut Down' button in the sidebar menu. This action closes the connection to the vector database and terminates the Streamlit session.

![Example Image](https://i.ibb.co/tXZZ4vz/shut-down.png)

---

## Additional informations

**Notes**

- Ensure you have a stable internet connection for using the RAG system, as it relies on the OpenAI API.
- For further assistance or inquiries, please refer to the project documentation or contact the developer.

**License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

**Acknowledgments**

This project was created by [bisd98](https://github.com/bisd98/Article-Retrieval-System).
