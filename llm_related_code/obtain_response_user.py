from sklearn.metrics.pairwise import cosine_similarity
import pickle
import asyncio
import aiohttp
import nest_asyncio
import json
import time
import ssl
import aiofiles
import os

# The api_key is hidden for anonymous review purposes. Please input your own api key.
api_key = "xxxxxx"
base_url = "https://api.deepseek.com/v1/chat/completions"

# indicate the dataset
file_path = './data/Beauty/'

# input request file
input_path = file_path + "user_prompt_input.pkl"

# output response file
write_path = file_path + 'responses_user_preference.json'  

if os.path.exists(write_path):
    os.remove(write_path)
    print(f"Deleted existing file: {write_path}")
    
with open(input_path, 'rb') as pickle_file:
    question_dic = pickle.load(pickle_file)


system_input = """Assume you are a beauty products recommendation expert.
You will be provided with a user's historical purchases of beauty products in chronological order, given in the following format:
<The title of item1>; <The title of item2>; <The title of item3>;... 
Please summarize the user's specific preference when purchasing beauty products. Note that your response should be a coherent paragraph of no more than 100 words.
"""



nest_asyncio.apply()


requests = [
    {"role": "user", "content": request, "item_key":key}
    for key, request in question_dic.items()
]


concurrent_limit = 200  
semaphore = asyncio.Semaphore(concurrent_limit)

batch_size = 1000

max_retries = 5

batch_delay = 5

ssl_context = ssl.create_default_context()

async def fetch_response(session, request, retries=0):
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_input},
                request,
            ],
            "temperature": 0.0,
            "top_p": 0.001,
            "stream": False
        }
        try:
            async with session.post(base_url, headers=headers, data=json.dumps(payload), ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=200)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Error: Received status code {response.status}")
                    return None
        except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorSSLError, asyncio.TimeoutError) as e:
            if retries < max_retries:
                print(f"Retrying... ({retries + 1}/{max_retries}) due to {e}")
                return await fetch_response(session, request, retries + 1)
            else:
                print(f"Failed after {max_retries} retries due to {e}.")
                return None

async def process_batch(session, batch):
    tasks = [fetch_response(session, request) for request in batch]
    return await asyncio.gather(*tasks)


async def write_responses_to_file(responses, file_path):
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(responses, indent=4))

async def main(file_path):
    start_time = time.time()  
    responses = {}  

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrent_limit, ssl=ssl_context)) as session:
        for i in range(0, len(requests), batch_size):
       
            print(f"Processing batch {i // batch_size + 1}")
            batch = requests[i:i + batch_size]
            batch_responses = await process_batch(session, batch)
            for request, response in zip(batch, batch_responses):
                item_key = request['item_key']
                responses[item_key] = response

          
            await write_responses_to_file(responses, file_path)
            print(f"Batch {i // batch_size + 1} completed, sleeping for {batch_delay} seconds")
            await asyncio.sleep(batch_delay)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Total time taken: {elapsed_time} seconds")
    return responses


responses = asyncio.run(main(write_path))
print(".......", len(list(responses.keys())), ".......")