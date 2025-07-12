#!/bin/bash

echo "🚀 Sales Agent Setup Script"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY environment variable is not set"
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY='your_api_key_here'"
    echo "  or create a .env file with GROQ_API_KEY=your_api_key_here"
else
    echo "✅ GROQ_API_KEY is set"
fi

# Check if MOCK_DATA.csv exists
if [ ! -f "MOCK_DATA.csv" ]; then
    echo "❌ MOCK_DATA.csv not found in current directory"
    echo "Please make sure the CSV file is in the same directory as this script"
    exit 1
fi

echo "✅ MOCK_DATA.csv found"

# Test the installation
echo "🧪 Testing installation..."
python3 test_sales_agent.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📖 Next steps:"
    echo "1. Set your Groq API key (if not already done):"
    echo "   export GROQ_API_KEY='your_api_key_here'"
    echo ""
    echo "2. Run the Streamlit app:"
    echo "   streamlit run sales_agent.py"
    echo ""
    echo "3. Or test programmatically:"
    echo "   python3 test_sales_agent.py"
    echo ""
    echo "🚀 Happy selling!"
else
    echo ""
    echo "❌ Setup test failed. Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "- Make sure GROQ_API_KEY is set correctly"
    echo "- Check internet connection (needed for embedding models)"
    echo "- Verify MOCK_DATA.csv is in the correct format"
fi 