# Biblical Insights AI - Intelligent Bible Assistant

A sophisticated AI-powered chatbot that provides detailed biblical insights, explanations, and interpretations using LangChain, Mistral AI, and Pinecone vector database.

## 🌟 Features

- Interactive chat interface with dynamic responses
- Context-aware biblical explanations
- Real-time verse references and interpretations
- Time-based greetings
- User-friendly interface with clear guidance
- Comprehensive biblical knowledge base
- Historical and contextual insights

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **Language Model**: Mistral AI
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace
- **Framework**: LangChain
- **Language**: Python 3.9+

## 📋 Prerequisites

- Python 3.9 or higher
- Mistral AI API key
- Pinecone API key
- 8GB RAM minimum
- Internet connection

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bible-bot.git
cd bible-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create `.streamlit/secrets.toml` for local development
   - Add to Streamlit Cloud for deployment

## 🔑 Configuration

Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your_mistral_key"
PINECONE_API_KEY = "your_pinecone_key"
PINECONE_ENVIRONMENT = "us-east-1"
```

## 🏃‍♂️ Running the Application

1. Local development:
```bash
streamlit run app.py
```

2. Access the application:
   - Local: http://localhost:8501
   - Deployed: Your Streamlit Cloud URL

## 📁 Project Structure

```
bible-bot/
├── app.py                 # Main application file
├── ui.py                  # UI components and layout
├── system_prompt.txt      # AI system instructions
├── requirements.txt       # Project dependencies
├── .streamlit/           
│   └── secrets.toml      # Local configuration
└── README.md             # Documentation
```

## 🔧 Core Components

1. **UI Layer (`ui.py`)**
   - Chat interface management
   - Session state handling
   - Dynamic greetings
   - User guide integration

2. **Application Core (`app.py`)**
   - LangChain integration
   - Vector store management
   - Query processing
   - Response generation

3. **System Prompt**
   - Response structuring
   - Biblical context handling
   - Tone and style guidelines

## 🎯 Features in Detail

### Chat Interface
- Real-time message updates
- Chat history management
- Clear conversation option
- Contextual responses

### AI Capabilities
- Scripture reference lookup
- Contextual understanding
- Verse interpretation
- Practical applications

### User Experience
- Time-based greetings
- Intuitive interface
- Helpful user guide
- Error handling

## 🔄 Workflow

1. **User Input**
   - Question submission
   - Input validation
   - Context preparation

2. **Processing**
   - Query refinement
   - Vector search
   - Context integration
   - Response generation

3. **Response**
   - Structured output
   - Biblical references
   - Practical insights
   - Additional resources

## 🛡 Security

- API keys stored in Streamlit secrets
- No sensitive data exposure
- Secure dependency versions
- Environment variable protection

## 📈 Performance

- Caching implementation
- Optimized vector searches
- Efficient response generation
- Minimal latency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

For support, please:
1. Check the documentation
2. Open an issue
3. Contact maintainers

## 🔮 Future Enhancements

- Multiple language support
- Advanced verse analysis
- Cross-reference functionality
- Historical context expansion
- Interactive study guides

## ⚠️ Important Notes

- Keep API keys secure
- Regular dependency updates
- Backup configuration files
- Monitor usage limits

## 🙏 Acknowledgments

- Streamlit team
- LangChain community
- Mistral AI team
- Pinecone team
- Open-source contributors
