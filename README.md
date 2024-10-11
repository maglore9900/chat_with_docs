This is a pretty simple python script that will allow you to chat with documents using OpenAI or Ollama. It supports command line arguments.

I made this tool because its one of the first things people try to do when they learn they can, and I hope it allows people to quickly see how to do it as well as improve upon it.

To use this tool you simply run it like so
`py main.py -d "test.pdf" -q "what is the first line in this document?"`

At some point I will finish adding the vectorstore options, but it currently doesnt work. Mostly just due to not having time to correct it.

# Install

so basically the steps are pretty simple

1. download the code (clone it or download it and unzip it)
2. install python 3.10 on the system
3. create a virtual environment using `python -m venv .` in the folder/dir of the code
4. activate the environment with `Scripts\activate.bat` on windows or `source bin/activate` on linux
5. run pip install to install all the required modules `pip install -r requirements_windows.txt`
6. then `cp example_env.txt to .env`
7. open that, and put in your info, like openai key or ollama or whatever
8. then run `python main.py` to start the whole thing up
