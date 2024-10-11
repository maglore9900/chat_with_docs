This is a pretty simple python script that will allow you to chat with documents using OpenAI or Ollama. It supports command line arguments.

I made this tool because its one of the first things people try to do when they learn they can, and I hope it allows people to quickly see how to do it as well as improve upon it.


To use this tool you simply run it like so
`py main.py -d "test.pdf" -q "what is the first line in this document?"`


At some point I will finish adding the vectorstore options, but it currently doesnt work. Mostly just due to not having time to correct it.

# Configuration

All of the easy configuration is done in the .env file. This section will explain what the values do, although you will also find it in the [example_env.txt](example_env.txt)

#LLM_TYPE will take openai, local. Local will use Ollama
`LLM_TYPE = 'openai'`

#-----OpenAI variables
`OPENAI_API_KEY = ''`
`OPENAI_MODEL = 'gpt-4o-mini'`

#-----Ollama variables
#OLLAMA_MODEL will take any model you can load in ollama
`OLLAMA_MODEL = 'gemma2'`
`OLLAMA_URL = 'http://localhost:11434'`
