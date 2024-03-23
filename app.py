import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import os


#Data Retrieval 

from apify_client import ApifyClient

# Initialize the ApifyClient with your API token
client = ApifyClient("apify_api_TBuXzkxhEamrvcvzhjhg22dcH02tjk2zlr7a")

# Prepare the Actor input




# for chainlit, .env is loaded automatically
from dotenv import load_dotenv
load_dotenv()  #
groq_api_key = "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"

# llm_local = ChatOllama(model="mistral:instruct")
llm_groq = ChatGroq(
            groq_api_key=groq_api_key,
            #model_name='llama2-70b-4096' 
            model_name='mixtral-8x7b-32768'
    )

@cl.on_chat_start
async def on_chat_start():
    
    files = None #Initialize variable to store uploaded files

    # Wait for the user to upload a file
    # while files is None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload a pdf file to begin!",
    #         accept=["application/pdf"],
    #         max_size_mb=100,
    #         timeout=180, 
    #     ).send()

    res = await cl.AskUserMessage(content="Provide the URL:", timeout=100).send()
    print(res['output'])

    
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing ...")
    msg.send()

    #Scraping the data and Standardising
    run_input = {
    "startUrls": [{ "url": res['output']}],
    "maxdepth":3,
    "onlyNewArticles": False,
    "onlyNewArticlesPerDomain": False,
    "onlyInsideArticles": True,
    "enqueueFromArticles": False,
    "crawlWholeSubdomain": False,
    "onlySubdomainArticles": False,
    "scanSitemaps": False,
    "saveSnapshots": False,
    "useGoogleBotHeaders": False,
    "minWords": 150,
    "mustHaveDate": True,
    "isUrlArticleDefinition": {
        "minDashes": 4,
        "hasDate": True,
        "linkIncludes": [
            "article",
            "storyid",
            "?p=",
            "id=",
            "/fpss/track",
            ".html",
            "/content/",
        ],
    },
    "proxyConfiguration": { "useApifyProxy": True },
    "useBrowser": False,
    "navigationWaitUntil": "load",
    "extendOutputFunction": """($) => {
    const result = {};
    // Uncomment to add a title to the output
    // result.pageTitle = $('title').text().trim();

    return result;
    }""",
    }
    run = client.actor("hy5TYiCBwQ9o8uRKG").call(run_input=run_input)
    articles={}

    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        articles[item['softTitle']]=item['text']
    # pdf = PyPDF2.PdfReader(file.path)
    text = ""
    for article,description in articles.items() :
        text+="Article:"+article+"\n"+"Description:"+str(description)+"\n"
        
    print(text)
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #embeddings = OllamaEmbeddings(model="llama2:7b")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing...  done. You can now ask questions!"
    await msg.update()
    #store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
        
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    #return results
    await cl.Message(content=answer, elements=text_elements).send()