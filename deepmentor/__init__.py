from deepmentor import config 

import os
import shutil
import requests
import json
import time
import base64
import random
from textwrap import dedent
from tqdm.auto import tqdm
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

# CrewAI Libs
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from crewai.flow.flow import Flow, listen, or_, and_, router, start

# Langchain Libs
from langchain_openai import ChatOpenAI
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain_classic.schema import HumanMessage

# Misc Libs
from PIL import Image
from IPython.display import display, Markdown
import openai
from pydantic import BaseModel, Field
import sqlite3
#import zipfile
import tiktoken # contagem de tokens
import asyncio

from typing import Any, Dict, List

# Warning controle
import warnings
warnings.filterwarnings('ignore')